# 3999f37 보존 배포판 검증

기준 커밋: `3999f3759fe6bfce1e62966bb38f0ce277add843`
브랜치: `preserve-3999f37-core` · 작업 경로: `/home/shchoi/PathBridger-preserved3999`

**p33·p44의 고정 입력 계산 동등성 기준을 통과했다.** 초기화·샘플러·Flow 난수 소비·분리 value 계산·bridge 원본 행/평균·update 순서를 이식했다. 코어 10개 파일 중 9개는 내부 import 경로 외에는 실제 원본 Git 객체와 바이트가 같다. 나머지 goal 표현 파일에는 기존 배포판의 scene 레이아웃 `(1,2,2)` 분기만 명시적 예외로 복원했다. 학습 loss·난수·update 수식은 수정하지 않았다. 현재 배포판의 config/CLI를 연결하고, CPU 공통 평가와 전체 상태 저장/재개를 적용했다.

## 학습 검증

- GPU, 배치 1024, seed 1. 초기화 전체 파라미터/Adam/EMA 및 원본 NumPy 샘플링을 비교했다.
- p33: 원본 100k 체크포인트의 **비영 Adam 상태**에서 16회 연속 비교. 모든 회차의 전체 loss/gradient, 가중치, Adam mu/nu/count, target EMA, 모델 RNG가 비트 단위로 동일하다. 최대 절대차는 모두 0이다.
- p44: 동일 초기화에서 3회 update한 비영 Adam 상태로부터 다시 16회 비교. 동일 항목이 모두 정확히 일치한다.
- 두 환경 모두 fresh 초기화에서 별도 3회 update도 일치한다. p33의 설정 매핑은 원본 실행 flags.json과 전체 키/값이 일치한다.
- 학습하지 않는 SPI actor는 만들지 않는다. 원본 action-Q/target-Q는 생성·학습·EMA·저장까지 유지하여 전체 계산 경로를 보존한다.
- 14개 PBF/PBG 설정은 변경하지 않은 원본과 비교했다. scene의 두 설정은 레이아웃 `(1,2,2)`만 고친 원본을 별도 기준으로 비교했다. 총 16개 설정이 초기화·샘플링·학습·저장/재개 검증을 통과했다. p33/p44 PBF와 scene은 fresh 3회 + continuation 16회, 나머지는 fresh 3회 + continuation 3회 검사다.

| 설정 | continuation update | 결과 |
|---|---:|---|
| p33 | 16 | bit exact |
| p44 | 16 | bit exact |
| pbf_antmaze_large | 3 | bit exact |
| pbf_antmaze_medium | 3 | bit exact |
| pbf_cube_double | 3 | bit exact |
| pbf_cube_single | 3 | bit exact |
| pbf_cube_triple | 3 | bit exact |
| pbg_antmaze_large | 3 | bit exact |
| pbg_antmaze_medium | 3 | bit exact |
| pbg_cube_double | 3 | bit exact |
| pbg_cube_single | 3 | bit exact |
| pbg_cube_triple | 3 | bit exact |
| pbg_puzzle_3x3 | 3 | bit exact |
| pbg_puzzle_4x4 | 3 | bit exact |
| pbf_scene | 16 | 레이아웃만 고친 원본과 bit exact |
| pbg_scene | 16 | 레이아웃만 고친 원본과 bit exact |

## 추론과 실행 인터페이스

- CPU 평가 형태인 batch 1 × 125개의 같은 observation·전체 goal·후보 RNG를 사용했다. 후보, ratio 점수, 선택 endpoint, bridge trajectory, raw action이 전부 동일하다. 공개 batched action API도 동일하다.
- reset observation과 **전체 goal**의 SHA-256이 기존 strict-reset 평가 기록 125개와 모두 일치한다.
- 공통 평가는 기존에 합의한 `3999-idm-strict-reset-v2`를 유지한다. 원본 평가 함수 body를 재사용하며, reset/Box 시드 고정 및 단계별 JIT 경계도 유지한다. 학습 seed가 평가 입력을 바꾸지 않는다.
- 실제 CLI: prefetch를 켜고 6 update를 연속 실행한 checkpoint와 3에서 저장 후 6까지 재개한 checkpoint 전체(호스트 RNG 포함)가 동일하다.
- 실제 CPU worker 두 평가 작업(각 task당 1 episode)은 GPU 학습과 별도 프로세스에서 완료했다. 기본 간격은 100k이며 학습은 CPU 평가 완료를 기다리지 않는다.
- 체크포인트는 config·차원·원본 두 agent·Adam·EMA·모델/호스트 RNG를 저장하고, CPU 평가는 학습 dataset 없이 복원한다. 설정 불일치 및 옛 unified 형식은 명확히 거부한다.
- CSV의 value/endpoint/bridge/IDM loss 이름을 유지했다. 원본 dynamics/critic loss도 함께 기록한다. critic total에는 Q loss가 추가되므로 이전 네 항목 합과 혼동하지 않아야 한다.

## Scene 복원 결과 (0.2.1)

- 관측 길이 40인 scene에 큐브 1개·버튼 2개·버튼당 상태 2개의 레이아웃을 명시했다. 원본의 조합 추정 오류를 피하고 기존 배포판과 같은 7차원 goal 표현을 만든다.
- OGBench 자체 oracle의 상태 210개·목표 10개와 정확히 일치한다. 이전 배포판과도 dataset 관측 1,024개가 정확히 일치한다.
- PBF와 PBG 각각 16회 실제 CLI 학습, 8/16 step 저장, 5개 task 각 1회 CPU 평가가 정상 완료됐다. 이는 실행 확인이며 성공률 재현 실험은 아니다.
- 두 scene 설정은 레이아웃만 고친 원본과 16회 loss/gradient·전체 가중치·Adam·EMA·난수 상태 및 저장/재개가 일치한다. 수정 전 원본은 scene을 초기화할 수 없으므로 이 조건을 명확히 구분한다.
- 수정 후 p33·p44의 16회 비교는 **변경하지 않은** 원본과 다시 일치했다. p33 CPU 추론 125건도 다시 통과했다.

실제 CLI의 짧은 학습 loss (4 → 16 update):

| 설정 | value | endpoint | bridge | IDM |
|---|---:|---:|---:|---:|
| PBF | 1.1908 → 0.5210 | 2.5472 → 1.5433 | 4.2554 → 2.9779 | 0.4756 → 0.1724 |
| PBG | 1.2165 → 0.5026 | 84.3942 → 61.5465 | 3.7300 → 2.9343 | 0.4758 → 0.1642 |

PBF의 endpoint flow MSE와 PBG의 endpoint Gaussian NLL은 서로 다른 지표이므로 절대 크기를 직접 비교하지 않는다.

## 속도·메모리

아래는 코어 이식 때의 0.2.0 측정값이다. 이번 변경은 scene 분기에 한정되며 p33/p44 연산 일치는 재검증했으나 속도 측정은 반복하지 않았다.

RTX 5090, driver 595.84, 동일 JAX/Flax/Optax, deterministic GPU ops + autotune 0. 각 경로를 별도 프로세스에서 10회 warmup 후 100 update씩 5회 측정했다. 아래 속도는 GPU에 올린 고정 batch의 update 처리량 중앙값이다. 컴파일, dataset IO, 저장, 평가 비용은 제외하며 샘플러 시간은 원시 JSON에 따로 기록했다. 선할당은 껐다.

| 환경 | 원본 update/s | 이식본 update/s | 처리량 변화 | 원본/이식본 peak live MiB | 원본/이식본 peak pool MiB |
|---|---:|---:|---:|---:|---:|
| p33 | 162.81 | 162.72 | -0.06% | 370.70 / 370.17 | 546 / 546 |
| p44 | 154.04 | 153.79 | -0.16% | 406.63 / 405.00 | 514 / 514 |

이 측정에서는 이식으로 인한 큰 비용이 관찰되지 않았다. 이 정도 처리량 차이를 일반적인 성능 차이로 해석하지 않는다. 학습 기본 JAX 선할당 옵션에서는 OS에 표시되는 GPU 예약 메모리가 위의 실제 allocator 사용량보다 클 수 있다.

## 확인된 범위와 제약

1. scene의 레이아웃 분기는 원본 그대로라는 주장에 대한 명시적 예외다. 수정 내용은 `PROVENANCE.json`과 실행별 `provenance.json`에 기록한다. 이전 보고서의 scene 지원 제외 판단을 철회하고 PBF/PBG 모두 지원한다.
2. 이전 unified 체크포인트에는 원본 Q/Adam 상태가 없어 정확한 학습 재개를 보장할 수 없다. 새 versioned checkpoint를 사용해야 한다.
3. 이 결과는 명시한 동일 runtime/device와 고정 입력에 대한 계산 검증이다. 다른 JAX/compiler/hardware에서 비트 동일함을 보증하지 않는다.
4. 추가 100k/1M 성능 학습이나 논문 전체 성공률 재현은 수행하지 않았다. 77.6% 대 71.2%의 기존 paired 성공 차이에 대한 새 인과 결론도 내리지 않는다.

## 산출물과 재실행

- `pathbridger_core/PROVENANCE.json`, `validation/git_source_verification.json`: 원본 Git 객체와 이식 파일 provenance.
- `validation/*equivalence.json`, `validation/inference_cpu.json`: 계산 비교 결과. `dataset_hashes.json`과 `runtime.json`에 입력 파일과 runtime을 기록했다.
- `validation/scene_layout.json`, `scene_runtime.json`, `scene_regressions.json`, `installed_scene.json`: scene 복원과 회귀 검증.
- `validation/runtime_smoke.json`, `package.json`, `installed_source_integrity.json`, `installed_commands.json`: CLI/저장/평가/배포 패키지 검증.
- `dist/pathbridger-0.2.1-py3-none-any.whl`: 배포 wheel. `.runtime-0.2.1/`에 별도 설치하여 현재 저장소 소스를 참조하지 않고 실행한다.
- 로컬 명령: `/home/shchoi/.local/bin/pathbridger-3999-train`, `/home/shchoi/.local/bin/pathbridger-3999-eval`.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true' \
python tests/check_core_equivalence.py \
  --reference-root /path/to/original3999/archive \
  --reference-run /path/to/original100k/run \
  --dataset /path/to/puzzle-3x3-play-v0-val.npz \
  --steps 16 --output validation/p33_equivalence.json
```

기존 작업 트리와 기존 학습 결과를 수정하지 않고 별도 브랜치/설치 경로에서 검증·배포했다.
