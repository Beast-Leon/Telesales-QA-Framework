# politeness_code
This repo contains retrained politeness model inspired by https://github.com/sudhof/politeness/tree/python3

# Structure of the repo.
1. Run generate_data.ipynb to generate serialized data
2. Run politeness_train.ipynb to train the customzied ML model.
3. Run test_model.ipynb to test out how the model perform.

# Update on Aug 10th
1. The first QA framework code is contained in QA_v1.ipynb
2. Need to retrain the politeness model to make sure politeness score is calculated correctly.

# Update on Aug 17th
1. Upload the third version QA frame work in QA_v3.ipynb
2. Use the fine-tuned distilbert sentiment model for testing the politeness. Currently all the model files are in local.
3. Use transformer sentence model for sentence encoding and category matching.
4. Need to enable multi-label category matching.
