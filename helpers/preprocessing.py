import re
import jieba
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator


class PreProcessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.selected_word = self.__get_dictionary(
            "data/list_temp_lemmed (1).csv", sep=" "
        )
        self.selected_word2 = self.__get_dictionary(
            "data/list_temp_lemmed.csv", sep="_"
        )

        self.__init_synonym_dict()

    def __init_synonym_dict(self):
        """Membaca file list_temp_lemmed yang merupakan list sinonim"""
        synonym_dict = {}
        for line in open("data/list_temp_lemmed.txt", "r"):
            words = line.strip().split(",")
            for i in range(1, len(words)):
                synonym_dict[words[i]] = words[0]

        self.synonym_dict = synonym_dict

    def __get_dictionary(self, file: str, sep: str) -> list:
        """Untuk membaca file list_temp_lemmed.csv
        dan list_temp_lemmed (1).txt

        Args:
            file (str): path filenya
            sep (str): pemisah/separator

        Returns:
            list: Isi file dalam bentuk list/array
        """
        syn1 = pd.read_csv(file, header=None, sep=sep, error_bad_lines=False)
        # Idk now what's going on here
        sin_direct2 = syn1.apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)

        for n, line in enumerate(sin_direct2):
            sin_direct2[n] = (
                " " + line.rstrip() if line.startswith("line") else line.rstrip()
            )
        return ",".join(sin_direct2).split(",")

    def pre_process_text(self, sentence: str) -> list:
        """Melakukan preprocessing kalimat, dimulai dari case folding
        menghapus angka, dan menghapus tanda baca kecuali koma, karena
        diasumsikan bahwa setiap gejala akan dipisahkan oleh koma

        Args:
            sentence (str): Satu kalimat yang akan dipreprocessing

        Returns:
            str: hasil preprocessing dalam bentuk word token
        """
        sentence = sentence.lower()  # case folding
        sentence = re.sub("\d+", "", sentence)  # remove numbers
        sentence = re.sub("/", " ", sentence)  # replace slash with space
        sentence = re.sub(" +", " ", sentence)  # remove double space
        sentence = sentence.split(",")
        sentence = [
            re.sub(r"[^\w\s]", "", symptom).strip() for symptom in sentence
        ]  # remove punctuation
        return sentence

    def remove_stopword(self, sentence: str) -> str:
        """Menghapus stopword dalam bahasa inggris

        Args:
            sentence (str): Satu kalimat

        Returns:
            str : Kalimat yang sudah dihapus stopwordnya
        """
        if sentence is None:
            return None

        word_token = nltk.word_tokenize(sentence)
        word_token = [word for word in word_token if not word in self.stopwords]

        return " ".join(word_token) if len(word_token) > 0 else None

    def get_selected_word(self, sentence: str, selected_word: list) -> str:
        """Menghapus kata-kata pada sentence yang tidak ada pada
        list/array `selected_word`

        Args:
            sentence (str): Kalimat yang akan dihapus kata-katanya
            selected_word (list): list kata-kata yang akan dipilih

        Returns:
            str: _description_
        """
        if sentence is None:
            return None

        word_token = nltk.word_tokenize(sentence)
        word_token = [word for word in word_token if word in selected_word]

        return " ".join(word_token) if len(word_token) > 0 else None

    def remove_none_in_list(self, list: list) -> list:
        """Menghapus None yang ada pada array/list"""
        return [word for word in list if word]

    def pos_tagger(self, nltk_tag, noun_to_verb=False):
        """Mengidentifikasi apakah kata tersebut
        verb, noun, atau adv

        Jika `noun_to_verb` bernilai True maka kata dalam bentuk
        noun akan dikembalikan kedalam bentuk verb
        """

        if nltk_tag.startswith("J"):
            return wordnet.VERB
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.VERB if noun_to_verb else wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        return None

    def tag_sentence(self, sentence: str, noun_to_verb=False) -> str:
        """Melakukan word tag menggunakan Wordnet pada satu kalimat

        Args:
            sentence (str): kalimat yang akan dilakukan tag
            noun_to_verb (bool, optional): Jika True maka kata noun akan diubah ke verb

        Returns:
            str: Kalimat hasil tag
        """
        words = []

        for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
            tag = self.pos_tagger(tag, noun_to_verb)
            word = self.lemmatizer.lemmatize(word, tag) if tag is not None else word
            words.append(word)

        return " ".join(words)

    def tag_sentences(self, sentences: list, noun_to_verb=False) -> list:
        """Melakukan word tag menggunakan Wordnet pada satu banyak kalimat sekaligus

        Args:
            sentences (list): List kalimat yang akan dilakukan tag
            noun_to_verb (bool, optional): Jika True maka kata noun akan diubah ke verb

        Returns:
            list: List kalimat hasil tag
        """
        return [self.tag_sentence(sentence, noun_to_verb) for sentence in sentences]

    def __jieba_split(self, sentence: str) -> list:
        """Kurang ngerti juga ini untuk apa"""
        # Without utf-8 encoding, it can't correspond to the words in the tongyici file.
        seg_list = jieba.cut(sentence, cut_all=False)
        return "/".join(seg_list).split("/")

    def tihuan_tongyici(self, sentence, with_jieba=False) -> str:
        """Ini kenapa namanya tihuan_tongyici?"""
        word_split = (
            self.__jieba_split(sentence) if with_jieba else sentence.split(", ")
        )

        final_sentence = ""
        for word in word_split:
            final_sentence += (
                self.synonym_dict[word] if word in self.synonym_dict else word
            )
        return final_sentence

    def translate_sentence(self, sentence: str) -> str:
        """Translate kalimat ke bahasa Inggris"""
        return GoogleTranslator(source="auto", target="en").translate(sentence)

    def translate_sentences(self, sentences: list) -> list:
        """Translate list kalimat ke bahasa inggris"""
        return [self.translate_sentence(sentence) for sentence in sentences]

    # 4 method dibawah bersifat private dan dibuat agar program lebih
    # rapi. Method-method ini hanya digunakan di method transform
    # Isi variabel sentances untuk 4 method dibawah adalah sebagai berikut
    # [
    #    ['stress', 'alcohol']       -> symptomps dari 1 orang
    #    ['symptom', 'symptom']      -> symptomps dari 1 orang
    # ]
    def __tag_sentence(self, sentences: list) -> list:
        word_tagged = [
            self.tag_sentences(symptoms, noun_to_verb=False) for symptoms in sentences
        ]
        return [
            self.tag_sentences(symptoms, noun_to_verb=True) for symptoms in word_tagged
        ]

    def __remove_stopword(self, sentences: list) -> list:
        temp = []
        for symptoms in sentences:
            temp.append([self.remove_stopword(symptom) for symptom in symptoms])
        return temp

    def __get_selected_word(self, sentences: list, selected_word: list) -> list:
        temp = []
        for symptoms in sentences:
            symptoms = [
                self.get_selected_word(symptom, selected_word) for symptom in symptoms
            ]
            temp.append(self.remove_none_in_list(symptoms))
        return temp

    def __tihuan_tongyici(self, sentences: list, with_jieba=False):
        temp = []
        for symptoms in sentences:
            temp.append(
                [self.tihuan_tongyici(symptom, with_jieba) for symptom in symptoms]
            )
        return temp

    def transform(self, sentences):
        """Method utama yang berfungsi untuk melakukan seluruh rangkaian
        preprocessing pada text gejala/symptoms

        Args:
            sentences (list or np.ndarray or str): List kalimat/kalimat yang akan dilakukan preprocessing.
                                                    Bisa diisikan satu kalimat atau banyak.

        Returns:
            list or str: hasil preprocessing
        """
        is_array = isinstance(sentences, np.ndarray) or isinstance(sentences, list)

        if not is_array:
            if not isinstance(sentences, str):
                raise Exception("Sentence harus berupa list, np.ndarray, atau string")
            sentences = [sentences]

        sentences = self.translate_sentences(sentences)
        self.translate = sentences
        sentences = [self.pre_process_text(symptoms) for symptoms in sentences]

        # mulai dari sini symptoms telah dipisah dalam bentuk array. Sehingga isi
        # dari variabel sentances seperti berikut ini
        # [
        #    ['stress', 'alcohol']       -> symptomps dari 1 orang
        #    ['symptom', 'symptom']      -> symptomps dari 1 orang
        # ]
        sentences = self.__tag_sentence(sentences)
        sentences = self.__remove_stopword(sentences)
        sentences = self.__get_selected_word(sentences, self.selected_word)
        sentences = self.__tihuan_tongyici(sentences)
        sentences = self.__tihuan_tongyici(sentences, with_jieba=True)
        sentences = self.__get_selected_word(sentences, self.selected_word2)

        # join
        return [" ".join(symptoms) for symptoms in sentences]
