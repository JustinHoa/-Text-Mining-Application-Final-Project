from datasets import load_dataset

ds = load_dataset("tranthaihoa/vifactcheck", split="train")

save_path = "seeding/"
ds.save_to_disk(save_path)