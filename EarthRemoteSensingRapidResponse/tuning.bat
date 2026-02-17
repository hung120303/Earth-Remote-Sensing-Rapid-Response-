
set LEARNING_RATES=0.1 0.01 0.001 0.0001
set WEIGHT_DECAYS=0.1 0.01 0.001 0.0001
set BATCH_SIZES=1 2 4 8 16
set EPOCHS=10 20 40 60 80 100
set PATCH_SIZES=4 8 16 32 64
set NUM_HEADS=4 8 16
set TRANSFORMER_LAYERS=4 8 12 24

for %%l in (%LEARNING_RATES%) do (
    for %%w in (%WEIGHT_DECAYS%) do (
        for %%b in (%BATCH_SIZES%) do (
            for %%e in (%EPOCHS%) do (
                for %%p in (%PATCH_SIZES%) do (
                    for %%n in (%NUM_HEADS%) do (
                        for %%t in (%TRANSFORMER_LAYERS%) do (
                            echo Training started for LR=%%l, WD=%%w, BS=%%b, EP=%%e, PS=%%p, NH=%%n, TL=%%t
                        )
                    )
                )
            )
        )
    )
)