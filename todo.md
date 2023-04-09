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
