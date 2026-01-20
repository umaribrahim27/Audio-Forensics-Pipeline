# Explainability Forensics Report

### Run Metadata
- Manifest: `/Users/seecsy/main/artifacts/manifests/test_800_balanced.csv`
- Limit analyzed: 200
- Model: `/Users/seecsy/main/checkpoints/bimodal_v2_explain/model.keras`
- Stats: `/Users/seecsy/main/artifacts/feature_stats/stats_train3000_v1.npz`

### Interpretation Notes
- **Time entropy**: low means explanations are concentrated in specific moments; high means spread across the clip.
- **Time concentration**: fraction of importance mass contained in the top 10% frames.
- **Freq entropy**: low means specific bands dominate; high means importance is spread across bands.

## ALL SAMPLES
- Count: 200
- p_fake mean/std: 0.2916 / 0.4389
- Time entropy mean/std: 5.8334 / 0.4244
- Time concentration mean/std: 0.2181 / 0.0554
- Freq entropy mean/std: 4.3447 / 0.3081
- Top mel bins (bin(count)): bin23(136), bin45(118), bin19(101), bin22(99), bin18(91), bin27(86), bin7(79), bin26(69), bin43(64), bin39(63)
- Top time frames (frame(count)): t1(96), t0(93), t2(76), t3(60), t4(47), t5(36), t6(24), t400(20), t7(19), t8(16)

## BONAFIDE (REAL)
- Count: 99
- p_fake mean/std: 0.0001 / 0.0005
- Time entropy mean/std: 5.8421 / 0.1191
- Time concentration mean/std: 0.2327 / 0.0651
- Freq entropy mean/std: 4.3674 / 0.0073
- Top mel bins (bin(count)): bin23(84), bin18(70), bin19(67), bin22(66), bin27(61), bin45(56), bin26(53), bin43(46), bin7(37), bin53(32)
- Top time frames (frame(count)): t0(56), t1(53), t2(41), t3(29), t4(21), t400(17), t5(14), t399(8), t6(6), t240(6)

## SPOOF (FAKE)
- Count: 101
- p_fake mean/std: 0.5774 / 0.4653
- Time entropy mean/std: 5.8248 / 0.5853
- Time concentration mean/std: 0.2038 / 0.0390
- Freq entropy mean/std: 4.3225 / 0.4324
- Top mel bins (bin(count)): bin45(62), bin23(52), bin7(42), bin6(39), bin19(34), bin22(33), bin39(31), bin37(26), bin55(25), bin27(25)
- Top time frames (frame(count)): t1(43), t0(37), t2(35), t3(31), t4(26), t5(22), t6(18), t7(15), t8(12), t9(10)

## Spoof vs Bonafide — Most Discriminative Mel Bins (by mean importance difference)
Top bins where spoof importance exceeds bonafide:
bin45(Δ=0.0604), bin6(Δ=0.0603), bin75(Δ=0.0579), bin7(Δ=0.0570), bin72(Δ=0.0564), bin37(Δ=0.0563), bin4(Δ=0.0562), bin3(Δ=0.0557), bin21(Δ=0.0548), bin55(Δ=0.0544)
