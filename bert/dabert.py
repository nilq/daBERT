from os.path      import join, dirname
from transformers import RobertaConfig, \
                         RobertaTokenizerFast, \
                         RobertaForMaskedLM, \
                         DataCollatorForLanguageModeling, \
                         Trainer, \
                         TrainingArguments, \
                         DataCollatorForLanguageModeling

from tokenizer import output_folder
import dataset

if __name__ == "__main__":
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(
        output_folder, max_len=512
    )

    model   = RobertaForMaskedLM(config=config)
    dataset = dataset.DanishDataset()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_folder,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_folder)