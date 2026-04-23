# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: llm
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 434.62 | 434.62 | 0.0 |
| Avg latency (ms) | 45983.25 | 47106 | 1122.75 |

## Failure modes
```json
{
  "by_agent": {
    "react": {
      "none": 8
    },
    "reflexion": {
      "none": 8
    }
  },
  "overall": {
    "none": 16
  },
  "correctness": {
    "correct": 16
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
