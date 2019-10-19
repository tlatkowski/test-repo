from datasets import coco
from models import txt2img_gan

def main():
    dataset, max_len, vocab_size = coco.coco_dataset_iterator('./annotations/captions_val2017.json', './val2017')
    gan = txt2img_gan.Text2ImageGAN(max_sentence_length=max_len, vocab_size=vocab_size)
    
    gan.fit(dataset)

if __name__ == '__main__':
    main()
