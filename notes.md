# 2023-04-17

* Implemented AdamW (after reading the original paper to get the intuition behind it)
* Next:
    * Implement a warmup: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
    * Not clear how they decay learning rate in BERT paper, original transformer paper used inverse square root steps

# 2023-04-10

* Cleaned up the code: made a `VanillaBert()` version so I can do a baseline test
* Took some code from Pytorch tutorial for positional encodings
* Next:
    * Add in AdamW (figure out how the L2 decay works), 
    * Implement a warmup: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
* BERT paper:
> We use Adam with learning rate of 1e-4, β1 = 0.9, β2 = 0.999, L2 weight
> decay of 0.01, learning rate warmup over the first 10,000 steps, and linear
> decay of the learning rate. We use a dropout prob- ability of 0.1 on all
> layers. We use a gelu acti- vation (Hendrycks and Gimpel, 2016) rather than
> the standard relu, following OpenAI GPT.

# 2023-04-10

* Got a transformer block working (courtesy of ChatGPT)
    * Tried to get ChatGPT to write me a test harness for the TransformerBlock and it hallucinated some datasets in Torch as well as some variables.  Didn't try to hard to get the prompt right, just decided to write my own.
    * I tried to ask it questions about the Pytorch/Lightning API, and it was semi-helpful.  What's the most helpful is if I can find the actual documentation, but it requires a few more clicks via Google/PyTorch docs
* Tested it on CoLA using the datamodule I had before
* Seems to be able to reduce training loss
* For more Layers (transformer blocks), I seem to have to reduce the learning rate
    * 1 Layer, I had to use LR =0.001
    * 2, 4, 8 Layer, I had to use LR =0.0001
    * 12 Layers, 0.0001 doesn't work well
* Validation loss doesn't seem to go, but I suspect the model can't really generalize enough from just the training dataset

# 2023-04-09

* Cramming builds off BERT base, which has:
    * Layers = 12
    * d_model = 768
    * heads = 12
    * head_dim = 768 / 12 = 64
    * d_ff = 4 * d_model = 3072
* BERT large:
    * L = 24
    * H=1024
    * A = 16 (heads)


# 2023-04-08

* Looks liked `DataCollatorForLanguageModeling`:
    * Batches stream into desired batch size (while padding if needed)
    * Adds labels using masked language model (15% or tunable words to be replaced), with 80% of those being masked, 10% random words, and 10% original
* Looks like I got a dataloader going that can do batches of 32 with MLM problem with 128 sequence length correctly
    * Speed is 100+ batches / second, which should be enough given my GPU
    * It hardly seems to tax CPU or disk/IO so shouldn't really be a bottleneck
    * Got to remember labels of -100 mean that you don't include it in the loss function

* `Side note`: It's sooooo useful for the code to all be open source.  You can easily inspect it (through Github or otherwise) and know exactly what's going on.


# 2023-04-06

* Incorporate `DataCollatorForLanguageModeling` into `data_scratch.ipynb` - `BertDataModule`
    * https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    * https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/data/data_collator.py#L607
* Get `BertDataModule` working
* Notes:
    * Not clear that collator will actually batch things for me, or will just pad them (read the source code)
    * Not clear if it expects `label` in the each row (or if it adds it in by itself)
