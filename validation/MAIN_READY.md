# main 반영 준비: 0.2.2

반영 대상은 `preserve-3999f37-core` 브랜치다. scene 지원을 유지한 상태에서 검증과 메모리 최적화를 마쳤다. 원본 코어와 추론 코드는 `3767154` 이후 변경하지 않았다.

- 구현 커밋: `e9db39cb4417626e48314dbaaa722fa9607c6e1c`
- 통합한 원격 main: `0bf1be462a063c66dc53a7367e0cfc2aabb3b5cf`
- 로컬 main: `e696ceeeb3928b025e66dcccb14a081f66da7389` (현재 위치 유지)
- 전체 검증: [REPORT.md](REPORT.md), [acceptance.json](acceptance.json).
- 설치본: `.runtime-0.2.2/`, wheel: `dist/pathbridger-0.2.2-py3-none-any.whl`.
- 기존 `pathbridger-3999-train` / `pathbridger-3999-eval` 로컬 명령은 검증된 0.2.2를 사용한다.

16개 설정의 연속 계산, p33/p44/scene 실제 CLI, loss 로그, 저장/재개, CPU 추론·평가, 설치본과 소스의 파일/실행 동등성이 통과했다. p33/p44에서 GPU peak live 메모리는 각각 9.1%/10.9% 감소했다. 처리량 변화는 +0.27%/-0.09%로 속도 향상을 주장하지 않는다. 예약 pool 크기는 같다. 전체 논문 성공률 재현은 별도 검증이다.

원격 main은 준비 브랜치의 조상이며 merge-tree 결과가 준비본과 같다. 따라서 강제 덮어쓰기 없이 fast-forward할 수 있다. main 작업 트리의 기존 untracked `logs/`, `scripts/`와 충돌하는 경로도 없다. 실제 main 반영과 원격 push는 아직 수행하지 않았다.

반영 시에는 원격 상태를 다시 확인하고 아래 순서로 실행한다. 원격 main에 새 커밋이 생겨 조상 검사가 실패하면 먼저 준비 브랜치에 통합하고 검증해야 한다.

```bash
cd /home/shchoi/PathBridger-paper
git fetch origin main &&
git merge-base --is-ancestor origin/main preserve-3999f37-core &&
git merge --ff-only preserve-3999f37-core &&
git push origin main
```

0.2.1 보존 코어 checkpoint는 그대로 재개할 수 있다. 더 오래된 unified checkpoint는 원본 Q/Adam 상태가 없어서 직접 재개할 수 없다. 버퍼 재사용만 끄려면 `--noreuse_train_buffers`를 사용한다.
