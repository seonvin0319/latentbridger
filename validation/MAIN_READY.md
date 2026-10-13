# main 반영 완료: 0.2.2

2026-10-13에 검증된 코드 `0eeeb8a0f203828d2e291e422e478942b5d68c85`를 main에 fast-forward로 반영하고 origin/main에 push했다. 통합된 작업 브랜치 `preserve-3999f37-core`는 삭제했다. 원격에는 해당 작업 브랜치가 없었다.

- 검증된 구현: `e9db39cb4417626e48314dbaaa722fa9607c6e1c`
- 검증 결과를 포함한 코드 기준: `0eeeb8a0f203828d2e291e422e478942b5d68c85`
- main 반영 전 원격 기준: `0bf1be462a063c66dc53a7367e0cfc2aabb3b5cf`
- 전체 검증: [REPORT.md](REPORT.md), [acceptance.json](acceptance.json).
- 반영 기록: [promotion.json](promotion.json). `main_integration.json`은 반영 준비 당시의 기록이다.

scene 지원과 현재 최적화를 유지했다. 16개 설정의 연속 계산, p33/p44/scene 실제 CLI, loss 로그, 저장/재개, CPU 추론·평가, 설치본과 소스의 동등성 검증을 통과했다. 논문 전체 성능 재현은 별도이며 PBF/PBG 모든 환경의 seed 0·1M 실험이 진행 중이다.

학습 큐는 검증된 `.runtime-0.2.2/` 설치본을 사용한다. 이 경로를 보존하기 위해 `/home/shchoi/PathBridger-preserved3999` 작업 폴더는 `0eeeb8a`의 detached HEAD로 유지했다. 브랜치 이름만 삭제했으며 학습 프로세스, 설치본, 결과 경로는 유지했다. 현재 실험이 사용하는 동안 이 작업 폴더를 제거하지 않는다.

일반 개발 작업 트리는 `/home/shchoi/PathBridger-paper`의 main이다. 기존 untracked `logs/`, `scripts/`는 유지했다. 0.2.1 보존 코어 checkpoint는 재개할 수 있지만, 더 오래된 unified checkpoint는 원본 Q/Adam 상태가 없어 직접 재개할 수 없다.
