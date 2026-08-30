(function (root) {
  "use strict";

  function simulatePlaybackTrace(trace, prebufferMs) {
    if (!trace || !Number.isInteger(trace.sampleRate) || trace.sampleRate <= 0) {
      throw new Error("sampleRate must be a positive integer");
    }
    if (!Array.isArray(trace.chunks) || trace.chunks.length === 0) {
      throw new Error("playback trace must contain chunks");
    }
    const selected = prebufferMs === undefined
      ? Number(trace.recommendedPrebufferMs)
      : Number(prebufferMs);
    if (!Number.isFinite(selected) || selected < 0) {
      throw new Error("prebufferMs must not be negative");
    }
    const rate = trace.sampleRate;
    const arrivals = trace.chunks.map(chunk => Math.round(Number(chunk.arrivalSeconds) * rate));
    let previous = -1;
    trace.chunks.forEach((chunk, index) => {
      if (!Number.isInteger(chunk.samples) || chunk.samples <= 0) {
        throw new Error("chunk samples must be positive integers");
      }
      if (arrivals[index] < 0 || arrivals[index] < previous) {
        throw new Error("chunk arrival times must be non-negative and monotonic");
      }
      previous = arrivals[index];
    });

    let playbackEnd = arrivals[0] + Math.round(selected * rate / 1000);
    let totalGapSamples = 0;
    const events = [];
    trace.chunks.forEach((chunk, index) => {
      const arrival = arrivals[index];
      const gapSamples = Math.max(0, arrival - playbackEnd);
      if (index > 0 && gapSamples > 0) {
        totalGapSamples += gapSamples;
        events.push({
          chunkIndex: index,
          arrivalSeconds: arrival / rate,
          bufferDepletedSeconds: playbackEnd / rate,
          gapSamples,
          gapSeconds: gapSamples / rate
        });
      }
      playbackEnd = Math.max(playbackEnd, arrival) + chunk.samples;
    });
    const largest = events.reduce((value, event) => Math.max(value, event.gapSamples), 0);
    return {
      prebufferMs: selected,
      underrunCount: events.length,
      underrunSamples: totalGapSamples,
      underrunSeconds: totalGapSamples / rate,
      largestUnderrunSamples: largest,
      largestUnderrunSeconds: largest / rate,
      gapEvents: events
    };
  }

  const SENTENCE_PUNCTUATION = new Set(Array.from(".!?。！？"));
  const CLAUSE_PUNCTUATION = new Set(Array.from(",;:、，；："));

  function isPunctuation(value) {
    return SENTENCE_PUNCTUATION.has(value) || CLAUSE_PUNCTUATION.has(value);
  }

  function splitGraphemes(text) {
    const source = String(text || "");
    if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
      const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
      return Array.from(segmenter.segment(source), item => ({
        text: item.segment,
        start: item.index,
        end: item.index + item.segment.length
      }));
    }
    let offset = 0;
    return Array.from(source, value => {
      const record = { text: value, start: offset, end: offset + value.length };
      offset = record.end;
      return record;
    });
  }

  function graphemeWeight(value) {
    if (/^\s+$/u.test(value)) return 0.22;
    if (SENTENCE_PUNCTUATION.has(value)) return 1.8;
    if (CLAUSE_PUNCTUATION.has(value)) return 1.05;
    if (/^[\p{Script=Latin}\p{Number}]$/u.test(value)) return 0.35;
    return 1.0;
  }

  function buildTextTimeline(text, language) {
    const source = String(text || "");
    const graphemes = splitGraphemes(source).map(item => ({
      ...item,
      weight: graphemeWeight(item.text),
      highlightable: !/^\s+$/u.test(item.text) && !isPunctuation(item.text)
    }));
    const totalWeight = graphemes.reduce((sum, item) => sum + item.weight, 0);
    let cursor = 0;
    const entries = graphemes.map(item => {
      const begin = totalWeight > 0 ? cursor / totalWeight : 0;
      cursor += item.weight;
      return {
        ...item,
        begin,
        endFraction: totalWeight > 0 ? cursor / totalWeight : 1
      };
    });
    const rates = { en: 5.2, ja: 4.8, zh: 4.8, yue: 4.6, ko: 5.0 };
    const rate = rates[String(language || "").toLowerCase()] || 4.8;
    const spoken = entries.filter(item => item.highlightable);
    const spokenWeight = spoken.reduce((sum, item) => sum + item.weight, 0);
    let spokenCursor = 0;
    const spokenEntries = spoken.map(item => {
      const begin = spokenWeight > 0 ? spokenCursor / spokenWeight : 0;
      spokenCursor += item.weight;
      return {
        ...item,
        begin,
        endFraction: spokenWeight > 0 ? spokenCursor / spokenWeight : 1
      };
    });
    return {
      text: source,
      entries,
      spokenEntries,
      totalWeight,
      estimatedDurationSeconds: Math.max(0.6, totalWeight / rate)
    };
  }

  function activeTextRange(timeline, progress) {
    const entries = timeline?.spokenEntries || timeline?.entries;
    if (!Array.isArray(entries) || !entries.length) return null;
    const position = Number(progress);
    if (!Number.isFinite(position) || position < 0 || position >= 1) return null;
    let low = 0;
    let high = entries.length - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const item = entries[middle];
      if (position < item.begin) high = middle - 1;
      else if (position >= item.endFraction) low = middle + 1;
      else return item.highlightable ? { start: item.start, end: item.end } : null;
    }
    return null;
  }

  function buildPcmActivityTimeline(chunks, sampleRate, options = {}) {
    const rate = Number(sampleRate);
    if (!Number.isInteger(rate) || rate <= 0) throw new Error("sampleRate must be positive");
    if (!Array.isArray(chunks) || !chunks.length) return null;
    const frameMs = Number(options.frameMs || 10);
    const thresholdDbfs = Number(options.thresholdDbfs || -50);
    const maximumBridgeMs = Number(options.maximumBridgeMs || 70);
    const frameSamples = Math.max(1, Math.round(rate * frameMs / 1000));
    const rms = [];
    const zeroCrossingRates = [];
    const roughness = [];
    let energy = 0;
    let absoluteDifference = 0;
    let zeroCrossings = 0;
    let samplesInFrame = 0;
    let previousSample = 0;
    let hasPreviousSample = false;
    function finishFrame() {
      if (!samplesInFrame) return;
      rms.push(Math.sqrt(energy / samplesInFrame));
      zeroCrossingRates.push(zeroCrossings / samplesInFrame);
      roughness.push(absoluteDifference / samplesInFrame);
      energy = 0;
      absoluteDifference = 0;
      zeroCrossings = 0;
      samplesInFrame = 0;
    }
    for (const chunk of chunks) {
      if (!chunk || !Number.isInteger(chunk.byteLength) || chunk.byteLength % 2) {
        throw new Error("PCM16 chunks must contain complete samples");
      }
      const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
      for (let offset = 0; offset < chunk.byteLength; offset += 2) {
        const value = view.getInt16(offset, true) / 32768;
        energy += value * value;
        if (hasPreviousSample) {
          absoluteDifference += Math.abs(value - previousSample);
          if ((value >= 0) !== (previousSample >= 0)) zeroCrossings += 1;
        }
        previousSample = value;
        hasPreviousSample = true;
        samplesInFrame += 1;
        if (samplesInFrame === frameSamples) finishFrame();
      }
    }
    finishFrame();
    if (!rms.length) return null;
    const threshold = 10 ** (thresholdDbfs / 20);
    const rawActive = rms.map(value => value > threshold);
    const active = rawActive.slice();
    const maximumBridgeFrames = Math.max(0, Math.round(maximumBridgeMs / frameMs));
    let previousActive = -1;
    for (let index = 0; index < active.length; index += 1) {
      if (!active[index]) continue;
      if (previousActive >= 0 && index - previousActive - 1 <= maximumBridgeFrames) {
        for (let fill = previousActive + 1; fill < index; fill += 1) active[fill] = true;
      }
      previousActive = index;
    }
    const first = active.indexOf(true);
    const last = active.lastIndexOf(true);
    if (first < 0 || last < first) return null;
    const activity = active.slice(first, last + 1);
    const rawActivity = rawActive.slice(first, last + 1);
    const slicedRms = rms.slice(first, last + 1);
    const slicedZeroCrossingRates = zeroCrossingRates.slice(first, last + 1);
    const slicedRoughness = roughness.slice(first, last + 1);
    const cumulativeActiveFrames = [0];
    for (const value of activity) {
      cumulativeActiveFrames.push(
        cumulativeActiveFrames[cumulativeActiveFrames.length - 1] + (value ? 1 : 0)
      );
    }
    const novelty = activity.map((isActive, index) => {
      if (!isActive || index === 0) return 0;
      const previousRms = Math.max(1e-7, slicedRms[index - 1]);
      const currentRms = Math.max(1e-7, slicedRms[index]);
      const energyChange = Math.min(
        3,
        Math.abs(20 * Math.log10(currentRms / previousRms)) / 12
      );
      const crossingChange = Math.min(
        3,
        Math.abs(
          slicedZeroCrossingRates[index] - slicedZeroCrossingRates[index - 1]
        ) * 8
      );
      const previousRoughness = Math.max(1e-7, slicedRoughness[index - 1]);
      const currentRoughness = Math.max(1e-7, slicedRoughness[index]);
      const roughnessChange = Math.min(
        3,
        Math.abs(20 * Math.log10(currentRoughness / previousRoughness)) / 12
      );
      return energyChange * 0.5 + crossingChange * 0.3 + roughnessChange * 0.2;
    });
    const nonzeroNovelty = novelty.filter(value => value > 1e-7).sort((a, b) => a - b);
    const noveltyScale = nonzeroNovelty.length
      ? nonzeroNovelty[Math.floor(nonzeroNovelty.length / 2)]
      : 1;
    const speechMotion = activity.map((isActive, index) => {
      if (!isActive) return 0;
      // A small base term lets sustained vowels advance. Acoustic changes carry
      // most of the progress, so character movement follows articulation rather
      // than a uniform wall-clock animation.
      return 0.18 + Math.min(3, novelty[index] / Math.max(1e-7, noveltyScale)) * 0.82;
    });
    const cumulativeSpeechMotion = [0];
    for (const value of speechMotion) {
      cumulativeSpeechMotion.push(
        cumulativeSpeechMotion[cumulativeSpeechMotion.length - 1] + value
      );
    }
    return {
      frameSeconds: frameSamples / rate,
      activity,
      rawActivity,
      novelty,
      cumulativeActiveFrames,
      activeFrameCount: cumulativeActiveFrames[cumulativeActiveFrames.length - 1],
      speechMotion,
      cumulativeSpeechMotion,
      totalSpeechMotion: cumulativeSpeechMotion[cumulativeSpeechMotion.length - 1],
      firstActiveSeconds: first * frameSamples / rate,
      lastActiveSeconds: Math.min(
        rms.length * frameSamples / rate,
        (last + 1) * frameSamples / rate
      )
    };
  }

  function punctuationStrength(value) {
    if (SENTENCE_PUNCTUATION.has(value)) return 2;
    if (CLAUSE_PUNCTUATION.has(value)) return 1;
    return 0;
  }

  function textSpeechGroups(timeline) {
    const groups = [];
    let spoken = [];
    for (const entry of timeline?.entries || []) {
      if (entry.highlightable) spoken.push(entry);
      const strength = punctuationStrength(entry.text);
      if (strength && spoken.length) {
        groups.push({ spoken, strength });
        spoken = [];
      }
    }
    if (spoken.length) groups.push({ spoken, strength: 0 });
    return groups;
  }

  function acousticPauses(acoustic, minimumPauseMs = 40) {
    const raw = acoustic?.rawActivity || acoustic?.activity || [];
    const minimumFrames = Math.max(
      1,
      Math.ceil((minimumPauseMs / 1000) / Math.max(1e-6, acoustic.frameSeconds))
    );
    const pauses = [];
    for (let index = 1; index < raw.length - 1;) {
      if (raw[index]) {
        index += 1;
        continue;
      }
      let end = index + 1;
      while (end < raw.length && !raw[end]) end += 1;
      if (end - index >= minimumFrames && end < raw.length) {
        pauses.push({ startFrame: index, endFrame: end, frames: end - index });
      }
      index = end;
    }
    return pauses;
  }

  function choosePauseAnchors(groups, acoustic) {
    if (groups.length < 2) return [];
    const pauses = acousticPauses(acoustic);
    const weights = groups.map(group => (
      group.spoken.reduce((sum, entry) => sum + Math.max(0.01, entry.weight), 0)
    ));
    const totalWeight = weights.reduce((sum, value) => sum + value, 0);
    const frameCount = acoustic.activity.length;
    let cumulative = 0;
    let previousPause = -1;
    const anchors = [];
    for (let boundary = 0; boundary < groups.length - 1; boundary += 1) {
      cumulative += weights[boundary];
      const expected = totalWeight > 0
        ? cumulative / totalWeight * frameCount
        : (boundary + 1) / groups.length * frameCount;
      let selected = -1;
      let selectedScore = Number.POSITIVE_INFINITY;
      for (let index = previousPause + 1; index < pauses.length; index += 1) {
        const pause = pauses[index];
        const center = (pause.startFrame + pause.endFrame) / 2;
        const strength = groups[boundary].strength || 1;
        const distance = Math.abs(center - expected) / Math.max(1, frameCount);
        const durationReward = Math.min(0.12, pause.frames / Math.max(1, frameCount));
        const score = distance / strength - durationReward;
        if (score < selectedScore) {
          selected = index;
          selectedScore = score;
        }
      }
      if (selected >= 0) {
        previousPause = selected;
        anchors.push(pauses[selected]);
      } else {
        const frame = Math.max(1, Math.min(frameCount - 1, Math.round(expected)));
        anchors.push({ startFrame: frame, endFrame: frame, frames: 0 });
      }
    }
    return anchors;
  }

  function snapBoundaryToArticulation(acoustic, expected, minimum, maximum, window) {
    const novelty = acoustic.novelty || [];
    const lower = Math.max(minimum, Math.floor(expected - window));
    const upper = Math.min(maximum, Math.ceil(expected + window));
    if (upper <= lower) return Math.max(minimum, Math.min(maximum, Math.round(expected)));
    let best = Math.max(lower, Math.min(upper, Math.round(expected)));
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let frame = lower; frame <= upper; frame += 1) {
      if (!acoustic.activity[frame]) continue;
      const distancePenalty = Math.abs(frame - expected) / Math.max(1, window);
      const score = Number(novelty[frame] || 0) - distancePenalty * 0.35;
      if (score > bestScore) {
        best = frame;
        bestScore = score;
      }
    }
    return best;
  }

  function alignTextTimelineToAudio(timeline, acoustic) {
    if (!acoustic || !Array.isArray(acoustic.activity) || !acoustic.activity.length) {
      return acoustic;
    }
    const groups = textSpeechGroups(timeline);
    if (!groups.length) return { ...acoustic, textAlignment: [] };
    const anchors = choosePauseAnchors(groups, acoustic);
    const aligned = [];
    let rangeStart = 0;
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
      const group = groups[groupIndex];
      const anchor = anchors[groupIndex];
      const rangeEnd = anchor ? anchor.startFrame : acoustic.activity.length;
      const entries = group.spoken;
      const totalWeight = entries.reduce(
        (sum, entry) => sum + Math.max(0.01, entry.weight),
        0
      );
      const span = Math.max(entries.length, rangeEnd - rangeStart);
      const boundaries = [rangeStart];
      let cumulative = 0;
      for (let index = 0; index < entries.length - 1; index += 1) {
        cumulative += Math.max(0.01, entries[index].weight);
        const expected = rangeStart + span * cumulative / Math.max(0.01, totalWeight);
        const minimum = boundaries[boundaries.length - 1] + 1;
        const maximum = Math.max(minimum, rangeEnd - (entries.length - index - 1));
        const searchWindow = Math.max(2, span / Math.max(4, entries.length * 2));
        boundaries.push(
          snapBoundaryToArticulation(
            acoustic,
            expected,
            minimum,
            maximum,
            searchWindow
          )
        );
      }
      boundaries.push(Math.max(boundaries[boundaries.length - 1] + 1, rangeEnd));
      entries.forEach((entry, index) => aligned.push({
        start: entry.start,
        end: entry.end,
        startFrame: boundaries[index],
        endFrame: boundaries[index + 1]
      }));
      rangeStart = anchor ? anchor.endFrame : rangeEnd;
    }
    aligned.sort((left, right) => left.startFrame - right.startFrame);
    return { ...acoustic, textAlignment: aligned };
  }

  function activeTextRangeAtAudioTime(timeline, acoustic, elapsedSeconds) {
    if (!acoustic || !acoustic.activeFrameCount) return null;
    const elapsed = Number(elapsedSeconds);
    if (!Number.isFinite(elapsed) || elapsed < 0) return null;
    const framePosition = elapsed / acoustic.frameSeconds;
    const frameIndex = Math.floor(framePosition);
    if (frameIndex < 0 || frameIndex >= acoustic.activity.length) return null;
    if (!acoustic.activity[frameIndex]) return null;
    if (Array.isArray(acoustic.textAlignment) && acoustic.textAlignment.length) {
      let low = 0;
      let high = acoustic.textAlignment.length - 1;
      while (low <= high) {
        const middle = (low + high) >> 1;
        const item = acoustic.textAlignment[middle];
        if (frameIndex < item.startFrame) high = middle - 1;
        else if (frameIndex >= item.endFrame) low = middle + 1;
        else return { start: item.start, end: item.end };
      }
      return null;
    }
    const fractionWithinFrame = Math.max(0, Math.min(1, framePosition - frameIndex));
    const hasSpeechMotion = (
      Array.isArray(acoustic.speechMotion)
      && Array.isArray(acoustic.cumulativeSpeechMotion)
      && acoustic.totalSpeechMotion > 0
    );
    const spokenProgress = hasSpeechMotion
      ? (
        acoustic.cumulativeSpeechMotion[frameIndex]
        + fractionWithinFrame * acoustic.speechMotion[frameIndex]
      ) / acoustic.totalSpeechMotion
      : (
        acoustic.cumulativeActiveFrames[frameIndex] + fractionWithinFrame
      ) / acoustic.activeFrameCount;
    return activeTextRange(timeline, Math.min(1 - Number.EPSILON, spokenProgress));
  }

  const api = {
    simulatePlaybackTrace,
    buildTextTimeline,
    activeTextRange,
    buildPcmActivityTimeline,
    alignTextTimelineToAudio,
    activeTextRangeAtAudioTime
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.AnifLiveTTSPlaybackModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
