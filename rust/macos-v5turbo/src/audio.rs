//! Native mono 48 kHz signed little-endian PCM16/WAV output.
use std::{error::Error, fs::File, io::Write, path::Path};
pub fn write_pcm16(wave: &[f32], wav_path: &Path, pcm_path: &Path) -> Result<(), Box<dyn Error>> {
    if wave.is_empty() || wave.iter().any(|v| !v.is_finite()) {
        return Err("Cannot encode empty or nonfinite audio".into());
    }
    let mut pcm = Vec::with_capacity(wave.len() * 2);
    for &sample in wave {
        pcm.extend_from_slice(&((sample * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes());
    }
    write_wav_bytes(&pcm, wav_path)?;
    File::create(pcm_path)?.write_all(&pcm)?;
    Ok(())
}
/// Package already-quantized streaming PCM without rescaling it.
pub fn write_pcm16_values(values: &[i16], path: &Path) -> Result<(), Box<dyn Error>> {
    let pcm: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_wav_bytes(&pcm, path)
}
fn write_wav_bytes(pcm: &[u8], wav_path: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = u32::try_from(pcm.len())?;
    let riff = bytes.checked_add(36).ok_or("WAV exceeds RIFF size limit")?;
    let mut wav = File::create(wav_path)?;
    wav.write_all(b"RIFF")?;
    wav.write_all(&riff.to_le_bytes())?;
    wav.write_all(b"WAVEfmt ")?;
    wav.write_all(&16u32.to_le_bytes())?;
    wav.write_all(&1u16.to_le_bytes())?;
    wav.write_all(&1u16.to_le_bytes())?;
    wav.write_all(&48000u32.to_le_bytes())?;
    wav.write_all(&96000u32.to_le_bytes())?;
    wav.write_all(&2u16.to_le_bytes())?;
    wav.write_all(&16u16.to_le_bytes())?;
    wav.write_all(b"data")?;
    wav.write_all(&bytes.to_le_bytes())?;
    wav.write_all(&pcm)?;
    Ok(())
}
