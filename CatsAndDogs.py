#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install -Uqq fastai duckduckgo_search')
from duckduckgo_search import ddg_images
from fastcore.all import * 
from fastai.vision.all import * 
from fastai.vision.widgets import * 


# In[12]:


def search_images(temp, max_images=300):
    print(f'Searching for {temp}')
    return L(ddg_images(temp, max_results=max_images)).itemgot('image')


# In[14]:


searches = 'Dog', 'Cat'
path = Path("Cats_and_Dogs")
from time import sleep

for i in searches:
    destination = (path/i)
    destination.mkdir(exist_ok=True, parents=True)
    download_images(destination, urls=search_images(f'{i} photo'))
    sleep(10)
    resize_images(path/i, max_sizes=400, destination = path/i)


# In[16]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# In[19]:


cats_dogs = DataBlock(
    blocks = (ImageBlock,CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = [Resize(128)],
    batch_tfms = aug_transforms(mult=2)
).dataloaders(path, bs=32)

cats_dogs.show_batch(max_n =10, unique=True)


# In[20]:


learn = vision_learner(cats_dogs,resnet18,metrics=error_rate)
learn.fine_tune(4)


# In[21]:


interpret = ClassificationInterpretation.from_learner(learn)
interpret.plot_confusion_matrix()


# In[22]:


interpret.plot_top_losses(5,nrows=5)


# In[24]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[25]:


for x in cleaner.delete():
    cleaner.fns[x].unlink()


# In[26]:


learn.export()


# In[27]:


ls


# In[ ]:




