# 2023-04-06

* Incorporate `DataCollatorForLanguageModeling` into `data_scratch.ipynb` - `BertDataModule`
    * https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    * https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/data/data_collator.py#L607
* Get `BertDataModule` working
* Notes:
    * Not clear that collator will actually batch things for me, or will just pad them (read the source code)
    * Not clear if it expects `label` in the each row (or if it adds it in by itself)
