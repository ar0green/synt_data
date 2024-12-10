from faker import Faker
import pandas as pd

def generate_text_data(num_samples=10, locale='en_US', text_type='sentence'):
    """
    Генерирует текстовые данные с помощью Faker.
    Параметры:
        num_samples: количество строк для генерации
        locale: локаль для Faker (например, 'en_US', 'ru_RU')
        text_type: тип генерируемого текста ('sentence', 'paragraph', 'name', etc.)
    """
    fake = Faker(locale)
    
    data = []
    for _ in range(num_samples):
        if text_type == 'sentence':
            data.append(fake.sentence())
        elif text_type == 'paragraph':
            data.append(fake.paragraph())
        elif text_type == 'name':
            data.append(fake.name())
        elif text_type == 'address':
            data.append(fake.address())
        else:
            # по умолчанию - предложение
            data.append(fake.sentence())
    
    df = pd.DataFrame(data, columns=["generated_text"])
    return df
