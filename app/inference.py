import os
import boto3
from pydantic import BaseModel
from typing import List

import tensorflow as tf
from fastapi import APIRouter, HTTPException

router = APIRouter()


class ItemList(BaseModel):
    title_list: List[str]


model_folder_name = './model/universal-sentence-encoder'


def load_model(model_folder_name):
    encoder = tf.keras.models.load_model(model_folder_name)
    return encoder


encoder = load_model(model_folder_name)


@router.post("/sentence_encoder")
def sentence_encoder(item: ItemList):
    try:
        score_list = encoder(item.title_list).numpy().tolist()
        return [{
            title: score
        } for title, score in zip(item.title_list, score_list)]
    except Exception as e:
        raise HTTPException(500, "sentence encoder fail :{}".format(e))
