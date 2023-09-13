# Dialouge ABSA
- For 2023.09.13
- Converted DiaASQ dataset for generative model (LLaMA-2) training.
- Notes
    - 1. Dataset statistics: train: valid = 800:100, since the other 100 is held out for competition (ended, but I have not updated)
    - 2. It is converted following the format discussed on 2023.09.12.
    - 3. For sentiment triplet appearance order, I use **opinion start index (in the full dialogue)** to sort ascendingly.
    - 4. For sentiment triplets lacking opinion spans, they are dropped.
    - 5. `doc_id` is the index coming with the original json files; serving as index.
