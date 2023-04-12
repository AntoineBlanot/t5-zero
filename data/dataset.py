import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets import load_dataset, Value


class DummyDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer, n: int, l: int) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = [
            dict(
                input_ids=torch.ones(l).long(),
                attention_mask=torch.ones(l).long(),
                label=1
            ) for _ in range(n)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __getitem__(self, index) -> dict:
        return self.data[index]
    

class MultiNLIDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer, do_tokenize: bool = True, do_prompt: bool = True) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("multi_nli", split=self.split)
        self.columns_to_keep = set(['input_ids', 'attention_mask', 'label'])

        if do_prompt:
            self.__prepare_prompt()
        else:
            self.__prepare_inputs()
        print('Data format before tokenizing: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))

        if do_tokenize:
            self.__tokenize()
        print('Data format after tokenizing: \n\t{}'.format(
                '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
            ))
        
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __prepare_prompt(self) -> None:
        def prompt_function(examples):
            input_text = [
                'hypothesis: {} {} premise: {} {} claim: {}'.format(
                    'The premise entaills the claim', self.tokenizer.eos_token, premise, self.tokenizer.eos_token, hypothesis
                )
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            prompt_function, batched=True, load_from_cache_file=False,
            desc='Preparing prompts for MNLI {} dataset'.format(self.split)
        )

    def __prepare_inputs(self) -> None:
        def inputs_function(examples):
            input_text = [
                '{} {} {}'.format(premise, self.tokenizer.sep_token, hypothesis)
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            inputs_function, batched=True, load_from_cache_file=False,
            desc='Preparing inputs for MNLI {} dataset'.format(self.split)
        )
    
    def __tokenize(self):
        preprocess_function = lambda examples: self.tokenizer(examples['input_text'], truncation=True)
        tokenized_data = self.data.map(
            preprocess_function, batched=True, load_from_cache_file=False,
            desc='Running tokenizer on MNLI {} dataset'.format(self.split)
        )
        self.data = tokenized_data.remove_columns(set(tokenized_data.column_names) - self.columns_to_keep)
    
    def __getitem__(self, index) -> dict:
        return self.data[index]


class BinaryNLIDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer, do_tokenize: bool = True, do_prompt: bool = True) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("multi_nli", split=self.split)
        def convert_binary(ex):
            ex['label'] = [1 if x == 0 else 0 for x in ex['label']]
            return ex
        self.data = self.data.map(
            convert_binary, batched=True, load_from_cache_file=False,
            desc='Convert MNLI {} dataset to binary dataset'.format(self.split)
        ).cast_column('label', Value('float32'))

        self.columns_to_keep = set(['input_ids', 'attention_mask', 'label'])

        if do_prompt:
            self.__prepare_prompt()
        else:
            self.__prepare_inputs()
        print('Data format before tokenizing: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))

        if do_tokenize:
            self.__tokenize()
        print('Data format after tokenizing: \n\t{}'.format(
                '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
            ))
        
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __prepare_prompt(self) -> None:
        def prompt_function(examples):
            input_text = [
                'hypothesis: {} {} premise: {} {} claim: {}'.format(
                    'The premise entaills the claim', self.tokenizer.eos_token, premise, self.tokenizer.eos_token, hypothesis
                )
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            prompt_function, batched=True, load_from_cache_file=False,
            desc='Preparing prompts for MNLI {} dataset'.format(self.split)
        )

    def __prepare_inputs(self) -> None:
        def inputs_function(examples):
            input_text = [
                '{} {} {}'.format(premise, self.tokenizer.sep_token, hypothesis)
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            inputs_function, batched=True, load_from_cache_file=False,
            desc='Preparing inputs for MNLI {} dataset'.format(self.split)
        )
    
    def __tokenize(self):
        preprocess_function = lambda examples: self.tokenizer(examples['input_text'], truncation=True)
        tokenized_data = self.data.map(
            preprocess_function, batched=True, load_from_cache_file=False,
            desc='Running tokenizer on MNLI {} dataset'.format(self.split)
        )
        self.data = tokenized_data.remove_columns(set(tokenized_data.column_names) - self.columns_to_keep)
    
    def __getitem__(self, index) -> dict:
        return self.data[index]


class ZeroShotDataset(Dataset):

    def __init__(self, split: str, files: list[str], tokenizer: AutoTokenizer, do_tokenize: bool = True, do_prompt: bool = True) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset('json', data_files=files, split='train')
        self.columns_to_keep = set(['input_ids', 'attention_mask', 'label', 'ref_list', 'group'])

        if do_prompt:
            self.__prepare_prompt()
        else:
            NotImplementedError()
        print('Data format before tokenizing: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))

        if do_tokenize:
            self.__tokenize()
        print('Data format after tokenizing: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __prepare_prompt(self) -> None:

        # def prompt_function(examples, indices):
        #     # BERT-MNLI (intent)
        #     input_text, label, ref_list, group = zip(*[
        #         ['{} {} {} question: {} answer: {}'.format(
        #             'The answer to the question is similar to:', convert_exemple(ref), self.tokenizer.sep_token,
        #             question, target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict
        
        def prompt_function(examples, indices):
            # T5-MNLI (intent)
            input_text, label, ref_list, group = zip(*[
                ['hypothesis: {} {} {} question: {} {} answer: {}'.format(
                    'The answer to the question is similar to:', convert_exemple(ref), self.tokenizer.eos_token,
                    question, self.tokenizer.eos_token, target
                ), label, ref_list, i]
                for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
                for ref in ref_list
            ])
            res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
            return res_dict
        
        
        # def prompt_function(examples, indices):
        #     # UniEval with question (intent)
        #     input_text, label, ref_list, group = zip(*[
        #         ['question: Is this a claim consistent with the premise? {} claim: {} {} {} premise: {} {}'.format(
        #             self.tokenizer.eos_token,
        #             'The answer to the question is similar to:', convert_exemple(ref), self.tokenizer.eos_token,
        #             question, target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict
        
        # def prompt_function(examples, indices):
        #     # UniEval without question (intent)
        #     input_text, label, ref_list, group = zip(*[
        #         ['question: Is this a claim consistent with the premise? {} claim: {} {} premise: {}'.format(
        #             self.tokenizer.eos_token,
        #             convert_exemple(ref), self.tokenizer.eos_token,
        #             target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict

        # def prompt_function(examples, indices):
        #     # UniEval with question (yes-no)
        #     input_text, label, ref_list, group = zip(*[
        #         ['question: Is the answer {} to the interrogation? {} interrogation: {} {} answer: {}'.format(
        #             convert_exemple(ref), self.tokenizer.eos_token,
        #             question, self.tokenizer.eos_token,
        #             target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict
        
        # def prompt_function(examples, indices):
        #     # UniEval with question (sentiment)
        #     input_text, label, ref_list, group = zip(*[
        #         ['question: Does the answer to the interrogation expresses {}? {} interrogation: {} {} answer: {}'.format(
        #             convert_exemple(ref), self.tokenizer.eos_token,
        #             question, self.tokenizer.eos_token,
        #             target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict
        
        # def prompt_function(examples, indices):
        #     # BART-MNLI (intent)
        #     input_text, label, ref_list, group = zip(*[
        #         ['{} {} {} question: {} answer: {}'.format(
        #             'The answer to the question is similar to:', convert_exemple(ref), self.tokenizer.sep_token,
        #             question, target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict

        # def prompt_function(examples, indices):
        #     # BART-MNLI (yes-no)
        #     input_text, label, ref_list, group = zip(*[
        #         ['The answer is {} to the question {} question: {} answer: {}'.format(
        #             convert_exemple(ref), self.tokenizer.sep_token,
        #             question, target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict

        # def prompt_function(examples, indices):
        #     # BART-MNLI (sentiment)
        #     input_text, label, ref_list, group = zip(*[
        #         ['The answer to the question expresses {} {} question: {} answer: {}'.format(
        #             convert_exemple(ref), self.tokenizer.sep_token,
        #             question, target
        #         ), label, ref_list, i]
        #         for i, question, target, ref_list, label in zip(indices, examples['question'], examples['answer'], examples['possible_intents'], examples['label'])
        #         for ref in ref_list
        #     ])
        #     res_dict = dict(input_text=list(input_text), label=list(label), ref_list=list(ref_list), group=list(group))
        #     return res_dict
        
        self.data = self.data.map(
            prompt_function, with_indices=True, batched=True, load_from_cache_file=False,
            desc='Preparing prompts for MNLI {} dataset'.format(self.split),
            remove_columns=self.data.column_names
        )
    
    def __tokenize(self):
        preprocess_function = lambda examples: self.tokenizer(examples['input_text'], truncation=True)
        tokenized_data = self.data.map(
            preprocess_function, batched=True, load_from_cache_file=False,
            desc='Running tokenizer on MNLI {} dataset'.format(self.split)
        )
        self.data = tokenized_data.remove_columns(set(tokenized_data.column_names) - self.columns_to_keep)

    def __getitem__(self, index) -> dict:
        return self.data[index]
    

def convert_exemple(name: str) -> str:
    """"
    Convert the intent name to a natural language sentence. Create an example.
    
    Params:
        - name: name of the intent to convert
    
    Returns:
        - converted intent
    """
    if name == 'yes':
        new_name = 'an agreement'
    elif name == 'no':
        new_name = 'a disagreement'
    
    elif name == 'positive':
        new_name = 'positivity'
    elif name == 'neutral':
        new_name = 'neutrality'
    elif name == 'negative':
        new_name = 'negativity'

    elif name == 'favorite-continent-asia':
        new_name = 'my favorite continent is Asia'
    elif name == 'favorite-continent-australia':
        new_name = 'my favorite continent is Australia'
    elif name == 'favorite-continent-africa':
        new_name = 'my favorite continent is Africa'
    elif name == 'favorite-continent-antarctica':
        new_name = 'my favorite continent is Antartica'
    elif name == 'favorite-continent-north-america':
        new_name = 'my favorite continent is North America'
    elif name == 'favorite-continent-europe':
        new_name = 'my favorite continent is Europe'
    elif name == 'favorite-continent-south-america':
        new_name = 'my favorite continent is South America'
    
    elif name == 'likes-tennis':
        new_name = 'my favorite sport is tennis'
    elif name == 'likes-baseball':
        new_name = 'my favorite sport is baseball'
    elif name == 'likes-basketball':
        new_name = 'my favorite sport is basketball'
    
    elif name == 'topic-books-likes-both-genre':
        new_name = 'I like reading both genre'
    elif name == 'topic-books-likes-fiction':
        new_name = 'I like reading fiction'
    elif name == 'topic-books-likes-non-fiction':
        new_name = 'I like reading non-fiction'

    elif name == 'topic-food-for-dinner':
        new_name = 'it is consumed for dinner'
    elif name == 'topic-food-for-breakfast':
        new_name = 'it is consumed for breakfast'
    elif name == 'topic-food-for-lunch':
        new_name = 'it is consumed for lunch'

    elif name == 'games':
        new_name = 'I like playing games'
    elif name == 'gardening':
        new_name = 'I like gardening'
    elif name == 'working-out':
        new_name = 'I like working-out'

    elif name == 'topic-hometown-big-city':
        new_name = 'it is a big city'
    elif name == 'topic-hometown-small-city':
        new_name = 'it is a small city'

    elif name == 'topic-profession-generic-profession':
        new_name = 'I have a generic profession'
    elif name == 'topic-profession-evil-profession':
        new_name = 'I have a profession related to evil'
    elif name == 'student':
        new_name = 'I am a student'

    elif name == 'topic-speaker-age-less-than-18-answer':
        new_name = 'I am still a child, less than 18 years old'
    elif name == 'topic-speaker-age-greater-than-18-answer':
        new_name = 'I am an adult, more than 18 years old'

    elif name == 'topic-travel-homecountry-human-from-india':
        new_name = 'I am from India'
    elif name == 'topic-travel-homecountry-human-from-japan':
        new_name = 'I am from Japan'
    elif name == 'topic-travel-homecountry-sarcastic-location':
        new_name = 'I am from a weird place'
    elif name == 'topic-travel-homecountry-human-from-usa':
        new_name = 'I am from the USA'
    elif name == 'topic-travel-homecountry-human-from-china':
        new_name = 'I am from China'

    elif name == 'favorite-season-summer':
        new_name = 'my favorite season is summer'
    elif name == 'favorite-season-winter':
        new_name = 'my favorite season is winter'
    elif name == 'favorite-season-spring':
        new_name = 'my favorite season is spring'
    elif name == 'favorite-season-autumn':
        new_name = 'my favorite season is autumn'

    elif name == 'watch-in-person':
        new_name = 'I like to watch it in person, where the action is'
    elif name == 'watches-on-tv':
        new_name = 'I like to watch it on TV'

    elif name == 'natural-wonders':
        new_name = 'I prefer natural wonders'
    elif name == 'man-made-monuments-answer':
        new_name = 'I prefer man-made monuments'

    elif name == 'topic-books-physical-books':
        new_name = 'I buy physical books wich are paper books'
    elif name == 'topic-books-ebooks':
        new_name = 'I buy ebooks which are digital books'

    elif name == 'topic-books-most-sold-book-rowling':
        new_name = 'J. K. Rowling sold more books'
    elif name == 'topic-books-most-sold-book-tolkien':
        new_name = 'J. R. R. Tolkien sold more books'

    elif name == 'topic-hometown-type-of-building-apartment-answer':
        new_name = 'I live in a appartment'
    elif name == 'topic-hometown-type-of-building-house-answer':
        new_name = 'I live in a house'

    elif name == 'topic-pet-eat-answer-little':
        new_name = 'my pet eats a little'
    elif name == 'topic-pet-eat-answer-lots':
        new_name = 'my pet eats a lot'

    elif name == 'topic-travel-homecountry-favorite-hemisphere-north':
        new_name = 'my favorite side of the globe is the north'
    elif name == 'topic-travel-homecountry-favorite-hemisphere-south':
        new_name = 'my favorite side of the globe is the south'

    elif name == 'like-to-play-sports':
        new_name = 'I like to play sports'
    elif name == 'like-to-watch-sports':
        new_name = 'I like to watch sports'

    elif name == 'topic-day-one-session-one-age-wrappingup-adulthood':
        new_name = 'I prefer adulthood'
    elif name == 'topic-day-one-session-one-age-wrappingup-childhood':
        new_name = 'I prefer chilhood'
    elif name == 'topic-day-one-session-one-age-wrappingup-oldage':
        new_name = 'I prefer old age'

    elif name == 'topic-language-learn-english-at-school':
        new_name = 'learn it at school'
    elif name == 'topic-language-learn-english-at-home':
        new_name = 'learn it at home'

    elif name == 'topic-birthday-days-february':
        new_name = '28 (twenty-eight) or 29 (twenty-nine) days'
    elif name == 'topic-birthday-days-thirty':
        new_name = '30 (thirty) days'
    elif name == 'topic-birthday-days-thirtyone':
        new_name = '31 (thirty-one) days'

    elif name == 'topic-day-three-food-noodles':
        new_name = 'I like noodles'
    elif name == 'topic-day-three-food-burgers':
        new_name = 'I like burgers'
    elif name == 'topic-day-three-food-pizza':
        new_name = 'I like pizza'

    elif name == 'topic-day-three-number-meals-between-three-six':
        new_name = 'between 3 and 6'
    elif name == 'topic-day-three-number-meals-greaterthan-six':
        new_name = 'greater than 6'
    elif name == 'topic-day-three-number-meals-lessthan-three':
        new_name = 'less than 3'

    elif name == 'likes-to-play-sports':
        new_name = 'I like to play'
    elif name == 'likes-to-watch-sports-or-fallback':
        new_name = 'I like to watch'

    elif name == 'topic-day-four-school-favorite-subject-science':
        new_name = 'my favorite subject is science'
    elif name == 'topic-day-four-school-favorite-subject-social':
        new_name = 'my favorite subject is socials'
    elif name == 'topic-day-four-school-favorite-subject-math':
        new_name = 'my favorite subject is mathematics'
    elif name == 'topic-day-four-school-favorite-subject-english':
        new_name = 'my favorite subject is english'
    
    elif name == 'topic-day-four-school-extra-curriculars-music':
        new_name = 'after shcool I play music'
    elif name == 'topic-day-four-school-extra-curriculars-sports':
        new_name = 'after school I play sports'
    
    elif name == 'topic-day-five-weather-rain':
        new_name = 'the rain is better'
    elif name == 'topic-day-five-weather-sun':
        new_name = 'the sun is better'

    elif name == 'topic-day-five-weather-favorite-season-summer':
        new_name = 'my favorite season is summer'
    elif name == 'topic-day-five-weather-favorite-season-winter':
        new_name = 'my favorite season is winter'
    elif name == 'topic-day-five-weather-favorite-season-fall':
        new_name = 'my favorite season is fall'
    elif name == 'topic-day-five-weather-favorite-season-spring':
        new_name = 'my favorite season is spring'

    elif name == 'topic-day-five-travel-sightseeing':
        new_name = 'I prefer sightseeing'
    elif name == 'topic-day-five-travel-food':
        new_name = 'I prefer the food'

    elif name == 'topic-olympics-select-user-would-compete-volleyball':
        new_name = 'I would compete in volleball'
    elif name == 'topic-olympics-select-user-would-compete-tennis':
        new_name = 'I would compete in tennis'
    elif name == 'topic-olympics-select-user-would-compete-diving':
        new_name = 'I would compete in diving'
    elif name == 'topic-olympics-select-user-would-compete-archery':
        new_name = 'I would compete in archery'

    elif name == 'topic-olympics-user-height-above-seventytwo':
        new_name = 'I am taller than 72 (seventy-two) inches'
    elif name == 'topic-olympics-user-height-below-seventytwo':
        new_name = 'I am less tall than 72 (seventy-two) inches'

    elif name == 'topic-olympics-haru-height-above-twentyfour':
        new_name = 'you are taller than 24 (twenty-four) inches'
    elif name == 'topic-olympics-haru-height-below-twentyfour':
        new_name = 'you are less tall than 24 (twenty-four) inches'

    else:
        print('`{}` not converted'.format(name))
        new_name = name

    return new_name