from pprint import pprint
import functools
import numpy as np


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from sklearn.metrics import confusion_matrix, f1_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm
from datasets import load_dataset


class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) et pas les poids du modèle 
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def training_step(self, batch):
        out = self.forward(batch)

        logits = out.logits
        # -------- MASKED --------
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est 
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

def xmlToDf(xmlFile):
    # Read XML file
    df = pd.read_xml(xmlFile)
    #replace None to empty string in commentaire column
    df["commentaire"] = df["commentaire"].apply(checkIfWordInComment)
    #check if not column exist
    if "note" in df.columns:
        # replace comma to point in note column
        df["note"] = df["note"].apply(lambda x: x.replace(",", "."))
        # string to double conversion column
        df['note'] = df['note'].astype(float)
        df['note'] = df['note'].apply(lambda x: x * 2 -1)
        df['note'] = df['note'].astype(int)
    return df


def checkIfWordInComment(comment):
    if comment is None:
        return ""
    return comment

df_train = xmlToDf("data/train.xml")
df_dev = xmlToDf("data/dev.xml")
df_test = xmlToDf("data/test.xml")


import json

# list all unique movies values of df_dev + df_train + df_test
movies = df_train["movie"].unique().tolist() + df_dev["movie"].unique().tolist() + df_test["movie"].unique().tolist()

#convert all movies values to int
for i in range(len(movies)):
    movies[i] = int(movies[i])
    
# load in a df all movies with json files in movies folder
df_movies = pd.DataFrame(columns=["name", "genre", "averageRate", "director"], index=movies)
for movie in movies:
    try:
        with open("movies/" + str(movie) + ".json", encoding="utf-8") as json_file:
            data = json.load(json_file, strict=False)
            # convert ratingValue to float by replacing comma to point and then multiply by 2 minus 1
            data['aggregateRating']['ratingValue'] = float(data['aggregateRating']['ratingValue'].replace(",", ".")) * 2 - 1           
            # if key director does not exist, we put an empty string
            if 'director' not in data:
                row = {'name': data['name'], 'genre': data['genre'], 'averageRate': data['aggregateRating']['ratingValue'], 'director': ""}
            elif type(data['director']) is list:
                row = {'name': data['name'], 'genre': data['genre'], 'averageRate': data['aggregateRating']['ratingValue'], 'director': data['director'][0]['name']}
            else:
                row = {'name': data['name'], 'genre': data['genre'], 'averageRate': data['aggregateRating']['ratingValue'], 'director': data['director']['name']}       
            df_movies.loc[movie] = row
    except:
        print("Error with movie " + str(movie))
        row = {'name': " ", 'genre': " ", 'averageRate': " ", 'director': " "}
        df_movies.loc[movie] = row


#define method to add special tokens to comment of a df row by looking for information in df_movies
def addSpecialTokens(row):
    movie = row["movie"]
    comment = row["commentaire"]
    # if movie is not in df_movies, we put empty string
    if movie not in df_movies.index:
        print("Error with movie " + str(movie))
        return "<UTILISATEUR>" + row["name"] + "<COMMENTAIRE>" + comment
    genre = df_movies.loc[int(movie)]["genre"]
    rate = df_movies.loc[int(movie)]["averageRate"]
    director = df_movies.loc[int(movie)]["director"]
    name = df_movies.loc[int(movie)]["name"]
    if type(genre) is list:
        genre = ";".join(genre)
    if type(row["name"]) is not str:
        return "<FILM>" + name + "<GENRE>" + genre + "<NOTE_MOYENNE>" + str(rate) + "<REALISATEUR>" + director + "<COMMENTAIRE>" + comment
    elif name == " ":
        return "<UTILISATEUR>" + row["name"] + "<COMMENTAIRE>" + comment
    else:   
        return "<FILM>" + name + "<GENRE>" + genre + "<NOTE_MOYENNE>" + str(rate) + "<REALISATEUR>" + director + "<UTILISATEUR>" + row["name"] + "<COMMENTAIRE>" + comment

#add special tokens to all comments
df_dev["commentaire_token"] = df_dev.apply(addSpecialTokens, axis=1)
df_train["commentaire_token"] = df_train.apply(addSpecialTokens, axis=1)

# convert df to dict
dict_train = df_train.to_dict('records')
dict_dev = df_dev.to_dict('records')
dict_test = df_test.to_dict('records')

def tokenize_batch(samples, tokenizer):
    text = [sample["commentaire_token"] for sample in samples]
    
    rates = [sample["note"] for sample in samples]
    labels = torch.tensor(rates).cuda()
    # The tokenizer handles
    # - Tokenization (amazing right?)
    # - Padding (adding empty tokens so that each example has the same length)
    # - Truncation (cutting samples that are too long)
    # - Special tokens (in CamemBERT, each sentence ends with a special token </s>)
    # - Attention mask (a binary vector which tells the model which tokens to look at. For instance it will not compute anything if the token is a padding token)
    tokens = tokenizer.batch_encode_plus( text,
                                        add_special_tokens=True,
                                        padding=True,
                                        truncation=True,
                                        max_length=512,
                                        return_attention_mask = True,
                                        return_tensors = 'pt')

    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "sentences": text}

tokenizer = AutoTokenizer.from_pretrained(
    'camembert/camembert-base',
    do_lower_case=True)

val_dataloader = DataLoader(
    dict_dev, 
    batch_size=8, 
    shuffle=False,
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer),
)
print("Preprocessing validation done")

train_dataloader = DataLoader(
    dict_train, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer),
)
print("Preprocessing train done")

# batch = next(iter(train_dataloader))
# print("\n".join(tokenizer.batch_decode(batch["input_ids"])))
# batch["labels"]

logger = TensorBoardLogger(name="camembert", save_dir="logs")

lightning_model = LightningModel("camembert-base", 10, lr=3e-5, weight_decay=0.)
model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
camembert_trainer = pl.Trainer(
    max_epochs=1,
    logger=logger,
    accelerator='gpu',
    devices=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]
)
camembert_trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
lightning_model = LightningModel.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

# save the model
torch.save(lightning_model.model.state_dict(), "model_meta_lightning.pt")
