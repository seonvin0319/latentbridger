# 3999f37 보존 배포판 검증

기준 커밋: `3999f3759fe6bfce1e62966bb38f0ce277add843`
브랜치: `preserve-3999f37-core` · 작업 경로: `/home/shchoi/PathBridger-preserved3999`

**p33·p44의 고정 입력 계산 동등성 기준을 통과했다.** 초기화·샘플러·Flow 난수 소비·분리 value 계산·bridge 원본 행/평균·update 순서를 이식했다. 코어 10개 파일은 내부 import 경로 외에는 실제 원본 Git 객체와 바이트가 같다. 현재 배포판의 config/CLI를 연결하고, CPU 공통 평가와 전체 상태 저장/재개를 적용했다.

## 학습 검증

- GPU, 배치 1024, seed 1. 초기화 전체 파라미터/Adam/EMA 및 원본 NumPy 샘플링을 비교했다.
- p33: 원본 100k 체크포인트의 **비영 Adam 상태**에서 16회 연속 비교. 모든 회차의 전체 loss/gradient, 가중치, Adam mu/nu/count, target EMA, 모델 RNG가 비트 단위로 동일하다. 최대 절대차는 모두 0이다.
- p44: 동일 초기화에서 3회 update한 비영 Adam 상태로부터 다시 16회 비교. 동일 항목이 모두 정확히 일치한다.
- 두 환경 모두 fresh 초기화에서 별도 3회 update도 일치한다. p33의 설정 매핑은 원본 실행 flags.json과 전체 키/값이 일치한다.
- 학습하지 않는 SPI actor는 만들지 않는다. 원본 action-Q/target-Q는 생성·학습·EMA·저장까지 유지하여 전체 계산 경로를 보존한다.
- 14개 사용 가능한 PBF/PBG 설정에서 초기화, 샘플러, loss/gradient, 반복 update, 저장/재개가 통과했다. p33/p44 PBF 이외는 fresh 3회 + continuation 3회의 짧은 검사다.

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

## 추론과 실행 인터페이스

- CPU 평가 형태인 batch 1 × 125개의 같은 observation·전체 goal·후보 RNG를 사용했다. 후보, ratio 점수, 선택 endpoint, bridge trajectory, raw action이 전부 동일하다. 공개 batched action API도 동일하다.
- reset observation과 **전체 goal**의 SHA-256이 기존 strict-reset 평가 기록 125개와 모두 일치한다.
- 공통 평가는 기존에 합의한 `3999-idm-strict-reset-v2`를 유지한다. 원본 평가 함수 body를 재사용하며, reset/Box 시드 고정 및 단계별 JIT 경계도 유지한다. 학습 seed가 평가 입력을 바꾸지 않는다.
- 실제 CLI: prefetch를 켜고 6 update를 연속 실행한 checkpoint와 3에서 저장 후 6까지 재개한 checkpoint 전체(호스트 RNG 포함)가 동일하다.
- 실제 CPU worker 두 평가 작업(각 task당 1 episode)은 GPU 학습과 별도 프로세스에서 완료했다. 기본 간격은 100k이며 학습은 CPU 평가 완료를 기다리지 않는다.
- 체크포인트는 config·차원·원본 두 agent·Adam·EMA·모델/호스트 RNG를 저장하고, CPU 평가는 학습 dataset 없이 복원한다. 설정 불일치 및 옛 unified 형식은 명확히 거부한다.
- CSV의 value/endpoint/bridge/IDM loss 이름을 유지했다. 원본 dynamics/critic loss도 함께 기록한다. critic total에는 Q loss가 추가되므로 이전 네 항목 합과 혼동하지 않아야 한다.

## 속도·메모리

RTX 5090, driver 595.84, 동일 JAX/Flax/Optax, deterministic GPU ops + autotune 0. 각 경로를 별도 프로세스에서 10회 warmup 후 100 update씩 5회 측정했다. 아래 속도는 GPU에 올린 고정 batch의 update 처리량 중앙값이다. 컴파일, dataset IO, 저장, 평가 비용은 제외하며 샘플러 시간은 원시 JSON에 따로 기록했다. 선할당은 껐다.

| 환경 | 원본 update/s | 이식본 update/s | 처리량 변화 | 원본/이식본 peak live MiB | 원본/이식본 peak pool MiB |
|---|---:|---:|---:|---:|---:|
| p33 | 162.81 | 162.72 | -0.06% | 370.70 / 370.17 | 546 / 546 |
| p44 | 154.04 | 153.79 | -0.16% | 406.63 / 405.00 | 514 / 514 |

이 측정에서는 이식으로 인한 큰 비용이 관찰되지 않았다. 이 정도 처리량 차이를 일반적인 성능 차이로 해석하지 않는다. 학습 기본 JAX 선할당 옵션에서는 OS에 표시되는 GPU 예약 메모리가 위의 실제 allocator 사용량보다 클 수 있다.

## 확인된 범위와 제약

1. `scene-play-v0`의 40차원 관측은 원본의 layout 추론에서 모호하다고 거부된다. PBF/PBG 양쪽에서 원본과 이식본이 같은 오류를 내는 것을 별도로 확인했다. 이 두 설정은 본 배포판에서 **사용 불가**이며, 원본을 임의로 수정하지 않았다.
2. 이전 unified 체크포인트에는 원본 Q/Adam 상태가 없어 정확한 학습 재개를 보장할 수 없다. 새 versioned checkpoint를 사용해야 한다.
3. 이 결과는 명시한 동일 runtime/device와 고정 입력에 대한 계산 검증이다. 다른 JAX/compiler/hardware에서 비트 동일함을 보증하지 않는다.
4. 추가 100k/1M 성능 학습이나 논문 전체 성공률 재현은 수행하지 않았다. 77.6% 대 71.2%의 기존 paired 성공 차이에 대한 새 인과 결론도 내리지 않는다.

## 산출물과 재실행

- `pathbridger_core/PROVENANCE.json`, `validation/git_source_verification.json`: 원본 Git 객체와 이식 파일 provenance.
- `validation/*equivalence.json`, `validation/inference_cpu.json`: 계산 비교 결과. `dataset_hashes.json`과 `runtime.json`에 입력 파일과 runtime을 기록했다.
- `validation/runtime_smoke.json`, `package.json`, `installed_source_integrity.json`, `installed_commands.json`: CLI/저장/평가/배포 패키지 검증.
- `dist/pathbridger-0.2.0-py3-none-any.whl`: 배포 wheel. `.runtime/`에 별도 설치하여 현재 저장소 소스를 참조하지 않고 실행한다.
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
