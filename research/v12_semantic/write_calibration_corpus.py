#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


LANGUAGE_SENTENCES = {
    "zh": [
        "今天的天气很好，我们出去走走吧。",
        "请慢一点说，我想听清楚每个字。",
        "新的系统已经准备好，可以开始测试了。",
        "灯光从窗边照进来，房间显得很温暖。",
        "这个答案很重要，请再确认一次。",
        "晚上回家以后，记得把门关好。",
        "我正在检查语音是否自然流畅。",
        "如果一切正常，我们就继续下一步。",
        "桌上的咖啡还热着，你要喝一点吗？",
        "谢谢你的耐心，今天辛苦了。",
    ],
    "yue": [
        "今日天氣幾好，我哋出去行吓啦。",
        "請你講慢少少，我想聽清楚每隻字。",
        "新系統已經準備好，可以開始測試喇。",
        "陽光由窗邊照入嚟，間房覺得好溫暖。",
        "呢個答案好重要，麻煩你再確認一次。",
        "夜晚返到屋企，記得閂好道門。",
        "我而家檢查緊把聲係咪自然流暢。",
        "如果全部正常，我哋就繼續下一步。",
        "枱面杯咖啡仲熱，你想唔想飲少少？",
        "多謝你咁有耐性，今日辛苦晒。",
    ],
    "en": [
        "The weather is pleasant today, so let us take a short walk.",
        "Please speak a little slower so I can hear every word clearly.",
        "The new system is ready, and we can begin the test now.",
        "Sunlight enters through the window and makes the room feel warm.",
        "This answer is important, so please check it one more time.",
        "Remember to close the door when you return home tonight.",
        "I am checking whether the speech sounds natural and smooth.",
        "If everything works correctly, we will continue to the next step.",
        "The coffee on the table is still warm; would you like some?",
        "Thank you for your patience and for all your work today.",
    ],
    "ja": [
        "今日はいい天気なので、少し散歩に出かけましょう。",
        "一つ一つの言葉が聞こえるように、少しゆっくり話してください。",
        "新しいシステムの準備ができたので、テストを始められます。",
        "窓から日差しが入り、部屋がとても暖かく感じられます。",
        "この答えは大切なので、もう一度確認してください。",
        "今夜家に帰ったら、忘れずにドアを閉めてください。",
        "音声が自然で滑らかに聞こえるか確認しています。",
        "すべて正常なら、次の手順に進みましょう。",
        "テーブルのコーヒーはまだ温かいですが、少し飲みますか？",
        "今日は最後まで付き合ってくれて、ありがとうございました。",
    ],
    "ko": [
        "오늘은 날씨가 좋아서 잠깐 산책을 나가도 좋겠습니다.",
        "모든 단어를 잘 들을 수 있도록 조금 천천히 말해 주세요.",
        "새 시스템이 준비되었으니 이제 테스트를 시작할 수 있습니다.",
        "창문으로 햇빛이 들어와서 방이 아주 따뜻하게 느껴집니다.",
        "이 답은 중요하니 한 번 더 확인해 주세요.",
        "오늘 밤 집에 돌아오면 문을 꼭 닫아 주세요.",
        "음성이 자연스럽고 매끄럽게 들리는지 확인하고 있습니다.",
        "모든 것이 정상이라면 다음 단계로 계속 진행하겠습니다.",
        "탁자 위의 커피가 아직 따뜻한데 조금 드시겠어요?",
        "오늘 끝까지 함께해 주셔서 정말 감사합니다.",
    ],
}


def build_rows(per_language: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language, sentences in LANGUAGE_SENTENCES.items():
        for index in range(per_language):
            first = sentences[index % len(sentences)]
            second = sentences[(index * 7 + 3) % len(sentences)]
            text = first if index < len(sentences) else f"{first} {second}"
            rows.append(
                {
                    "id": f"{language}-{index:04d}",
                    "language": language,
                    "text": text,
                    "seed": seed + len(rows) * 1009,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.per_language < 1:
        parser.error("--per-language must be positive")
    rows = build_rows(args.per_language, args.seed)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(target), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
