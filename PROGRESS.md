# 진행 상황 (Netflix 고객 이탈 분석)

## 단계
- [x] 0. 프로젝트 세팅 (공통 설정 복사, 원격 ych830/netflix-churn)
- [x] 1. EDA → `reports/01_eda.md`, `src/eda.py`
- [x] 2. 기획 → `reports/02_plan.md` (목적 C: 예측+전략, 산출물은 포트폴리오 보고서로 가정)
- [x] 3. 전처리·피처 정리 (`monthly_fee`·`avg_watch` 제외, 인코딩은 model.py 파이프라인)
- [x] 4. 모델링 → `reports/03_model.md`. 라벨 규칙 역추정 `src/risk_score.py` (점수>=3, 96.5%)
- [ ] 5. 해석: 변수 중요도, 고객 세그먼트, 임계값 규칙
- [ ] 6. 결과물: 리포트 / 대시보드 / 액션 제안

## 다음 할 일
- 5단계: 위험 점수대 × 주요 원인별 세그먼트 표, 세그먼트별 액션 제안 (`src/segment.py`, `reports/04_segments.md`)
- 산출물 형태·대상 사용자 확정 필요 (현재 포트폴리오 보고서로 가정)
