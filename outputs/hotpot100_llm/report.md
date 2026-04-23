# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: llm
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 430.38 | 430.38 | 0.0 |
| Avg latency (ms) | 13151.96 | 13244.15 | 92.19 |

## Failure modes
```json
{
  "by_agent": {
    "react": {
      "none": 120
    },
    "reflexion": {
      "none": 120
    }
  },
  "overall": {
    "none": 240
  },
  "correctness": {
    "correct": 240
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
