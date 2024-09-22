# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/21 18:15
@Auth ： max
@File ：Advanced.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import random
import json
import hashlib
from typing import List, Dict, Any, Optional
import datetime
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageChops
import os
import base64
import io
import logging
from logging.handlers import RotatingFileHandler
import sys

IMAGE_DATA = "iVBORw0KGgoAAAANSUhEUgAAADgAAAA4CAMAAACfWMssAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKFUExURQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACYmJgAAAAAAAB4eHgAAAB4eHh0dHQAAABwcHBwcHBgYGBgYGBcXFxYWFgAAABYWFi0tLSwsLCsrKygoKCcnJzs7OyMjIzY2NiAgIDMzMzIyMjAwMDAwMBwcHC0tLSwsLCkpKT09PTk5OTQ0ND4+Pjo6Ojk5OTk5ORERETc3NzU1NRkZGTQ0NEJCQjAwMEBAQD4+PiIiIjo6Ojo6OigoKDY2NiUlJUBAQEFBQT8/Py8vLzw8PDw8PENDQ0FBQVNTU0BAQD4+Pj8/P1BQUD4+PkhISF9fX0dHR0JCQkNDQ1dXV3JyclRUVFVVVW9vb1NTU21tbWJiYlxcXFlZWXp6enV1dXZ2dm1tbW5ubm9vb2pqamNjY4WFhYCAgIKCgnp6enx8fH19fXR0dHV1dXd3d3h4eBQUFHJycnR0dG5ubklJSVxcXJOTk1dXV4eHh4qKik9PT4CAgH5+fiQkJGFhYSgoKFdXV1xcXLS0tKWlpZubm56enoqKio2NjZGRkZOTk319fYWFhY6OjpCQkJWVlZycnJ+fn6GhoaKioqmpqaqqqq+vr7CwsLGxsbW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb+/v8DAwMLCwsnJyc3Nzc/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tnZ2dra2tvb29zc3N3d3d/f3+Dg4OHh4ePj4+Tk5OXl5ebm5ufn5+jo6NQN6h8AAACldFJOUwABAwQGBwkMDQ4PFRYcHSMkJSgpKissLi8xMjM0NTY3OEBCREhKS1FUVVVZWltcXmFlZ2lpbG1wdHV6fH+Ag4SFhoiJio+PlJmgpKWmqKipq6utr6+wsrS1t7e8vLy9vr/AxsfHyMnJycrNzc7Q0NDQ0dHR0tLW2NnZ2trc3Nzd3+Hi4uPj4+Tk5OTl5eXm5+rq6+vr7Ozt7vDx8vLy8/T09fX19eGp09YAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAKFSURBVEhL7db7X0txHMfxbzOlJYpoZq6506jc7/dbKeXSkDsdklhyv2QMbe7kfonkXolilCTJsGHV+vw9vmf7nLXOpa0HP9Hzp+18X+/H4zx2e4z8f1SDJakwEae68ELCDS/D4yDh8t8eRrjESA9jMMEBR1uQ54KdwHPXcYEWBxytDQMvbCk44PzJ8GU6dR8DgWs6nW7fG/HhOLm8j+SLY+itUEySGI5p6e0wdCVkfNtQnJdht4PYCRzpLD5MtjqHHTdhJ5ClIGTRU7CuxgFn4UeomEaI31YHhjw/02g0/T1UJ7l6t9GlYGHvYspbLHnK5tLDZAuUTXDmTfrlAlztRV8dI5Y8h8MJ6XkA4AH/++if+gPK4/0JiXuMaTOFSwhpF2sGOxOIA7fohwB59D5Ue60Ye/iWqSQk6iLAo8mYNwlaXwVwMpqQYafsmLvZc4YTMmo3QPX2YMw99M2mSe6sTiTSWOfqOXXGSBI0dX8jwNmBGDcTda4eoCR9RIjG1IATp3qTJliz+S59lL8YU56RZ74CfC/es2bVFY+l49by+C33aulJzlgMBSIyX7Gp9cOTQvfnwPGs6N0X9kHp0SGYiQiZsc3c7DbdKg7N7oKRuO7zUg3FVZZf2LNsn0qOpc3pgYG09uFDFyzbsHGHmR2Vn8/KSEmaqO6Ah97J+5vYoWFQoB9e8RlD35yGDHzSGnGfAWqX4pPWiK0BqOF/+3zRNpQQoFSrV7DDtWq1MgAv+kC58oRer2eH+Xr9znX0Z8MXMpksoZL9uHEqE+glPJQWOv80w1zHCbrJMMzMUAwkyBPv0J8WocbXiXJMxIXtwlLAFIaJONWlIgm3vfyXGyCp5eE/iJDffBZ6h8vn7WsAAAAASUVORK5CYII="

SECRET_KEY = "DgAAAA4CAMAAACfWMssAAAAAXNSR0IArs4c6QAAAARnQU1BAACx"

# 设置日志记录器
logger = logging.getLogger('CatBreeder')
logger.setLevel(logging.DEBUG)

# 创建一个 RotatingFileHandler
file_handler = RotatingFileHandler('run.log', maxBytes=1024 * 1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(file_handler)


# 捕获未处理的异常
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def get_image():
    image_data = base64.b64decode(IMAGE_DATA)
    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGBA")


def calculate_checksum(data):
    return hashlib.md5((data + SECRET_KEY).encode()).hexdigest()


class Personality:
    def __init__(self, trait: str, value: int):
        self.trait = trait
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"trait": self.trait, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Personality':
        return cls(data["trait"], data["value"])


class Cat:
    def __init__(self, name, color, pattern, personalities, age, is_female, parents=None):

        self.name = name
        self.color = color
        self.pattern = pattern
        self.personalities = personalities
        self.age = age
        self.is_female = is_female
        self.parents = parents or []
        self.is_sick = False
        self.sick_days = 0
        self.recovery_chance = 0.5  # 50% chance to recover each year
        self.is_neutered = False
        self.breeding_end_age = random.randint(15 * 12, 20 * 12)  # 随机设定繁殖结束年龄，在15-20年之间

    def age_display(self):
        years = self.age // 12
        months = self.age % 12
        if years == 0:
            return f"{months}个月"
        elif months == 0:
            return f"{years}岁"
        else:
            return f"{years}岁{months}个月"

    def get_sick(self):
        if not self.is_sick:
            self.is_sick = True
            self.sick_days = 0

    def update_personality(self, change_rate):
        for personality in self.personalities:
            change = random.uniform(-change_rate, change_rate)
            personality.value = max(1, min(10, personality.value + change))

    def try_recover(self):
        if self.is_sick:
            self.sick_days += 1
            if random.random() < self.recovery_chance:
                self.heal()
                return True
        return False

    def heal(self):
        self.is_sick = False
        self.sick_days = 0

    def meow(self) -> str:
        loudness = sum(p.value for p in self.personalities if p.trait == "Energetic")
        shyness = sum(p.value for p in self.personalities if p.trait == "Shy")

        meow_types = [
            "^..^",  # 非常安静
            "mew",  # 轻声
            "meow",  # 普通
            "MEOW",  # 大声
            "MEOW!",  # 非常大声
        ]

        meow_index = min(4, max(0, int((loudness - shyness + 10) / 5)))
        meow = meow_types[meow_index]

        if random.random() < 0.2:  # 20% 的概率重复叫声
            meow = f"{meow} {meow}"

        return meow

    def can_breed(self):
        return 12 <= self.age < self.breeding_end_age and not self.is_neutered

    def update_age(self):
        self.age += 1
        if self.age >= self.breeding_end_age:
            self.is_neutered = True

    def age_up(self, personality_change_rate: float):
        self.age += 1
        for personality in self.personalities:
            change = random.uniform(-personality_change_rate, personality_change_rate)
            personality.value = max(1, min(10, personality.value + change))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "pattern": self.pattern,
            "personalities": [p.to_dict() for p in self.personalities],
            "age": self.age,
            "is_female": self.is_female,
            "parents": self.parents
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cat':
        return cls(
            data["name"],
            data["color"],
            data["pattern"],
            [Personality.from_dict(p) for p in data["personalities"]],
            data["age"],
            data["is_female"],
            data["parents"]
        )


class CatNameGenerator:
    def __init__(self):
        self.prefixes = ["Mr", "Mrs", "Sir", "Lady", "Little", "Big", "Old", "Young", "Captain", "Professor"]
        self.base_names = ["Whiskers", "Mittens", "Fluffy", "Oreo", "Luna", "Tiger", "Smokey", "Mochi", "Neko"]
        self.suffixes = ["Jr", "II", "III", "the Great", "the Wise", "the Brave", "the Cute"]
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population = self.initialize_population()

    def initialize_population(self):
        population = self.base_names.copy()
        while len(population) < self.population_size:
            new_name = self.generate_random_name()
            if new_name not in population:
                population.append(new_name)
        return population

    def generate_random_name(self):
        name_parts = []
        if random.random() < 0.3:
            name_parts.append(random.choice(self.prefixes))

        if random.random() < 0.7:
            name_parts.append(self.generate_word())
        else:
            name_parts.append(random.choice(self.base_names))

        if random.random() < 0.3:
            name_parts.append(random.choice(self.suffixes))

        return " ".join(name_parts)

    def generate_word(self):
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        length = random.randint(3, 8)
        word = ''
        for i in range(length):
            if i % 2 == 0:
                word += random.choice(consonants)
            else:
                word += random.choice(vowels)
        return word.capitalize()

    def fitness(self, name):
        score = 0
        words = name.split()
        score += len(words)  # 更长的名字得分更高
        score += len(set(name.lower())) / len(name)  # 独特字符比例
        if any(prefix in name for prefix in self.prefixes):
            score += 1
        if any(suffix in name for suffix in self.suffixes):
            score += 1
        return score

    def select_parent(self):
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.fitness)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            words1 = parent1.split()
            words2 = parent2.split()

            # 处理父本名字只有一个单词的情况
            if len(words1) <= 1 or len(words2) <= 1:
                return self.generate_random_name(), self.generate_random_name()

            split1 = random.randint(1, len(words1) - 1)
            split2 = random.randint(1, len(words2) - 1)
            child1 = " ".join(words1[:split1] + words2[split2:])
            child2 = " ".join(words2[:split2] + words1[split1:])
            return child1, child2
        return parent1, parent2

    def mutate(self, name):
        if random.random() < self.mutation_rate:
            words = name.split()
            index = random.randint(0, len(words) - 1)
            if random.random() < 0.5:
                words[index] = self.generate_word()
            else:
                if index == 0 and random.random() < 0.3:
                    words.insert(0, random.choice(self.prefixes))
                elif index == len(words) - 1 and random.random() < 0.3:
                    words.append(random.choice(self.suffixes))
            return " ".join(words)
        return name

    def evolve(self):
        new_population = []
        elite_size = int(0.1 * self.population_size)
        elite = sorted(self.population, key=self.fitness, reverse=True)[:elite_size]
        new_population.extend(elite)

        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))

        self.population = new_population

    def get_name(self):
        return random.choice(self.population)


class CatGenerator:
    def __init__(self, breeder):
        self.breeder = breeder
        self.name_generator = CatNameGenerator()
        self.colors = ["Black", "White", "Orange", "Gray", "Brown"]
        self.patterns = ["Solid", "Tabby", "Calico", "Tortoiseshell", "Tuxedo", "Spotted"]
        self.personality_traits = ["Shy", "Energetic", "Lazy", "Curious", "Friendly"]
        self.male_prefixes = ["Mr", "Sir", "Lord", "Duke"]
        self.female_prefixes = ["Mrs", "Lady", "Queen", "Duchess"]

    def generate_name(self, is_female):
        prefix = random.choice(self.female_prefixes if is_female else self.male_prefixes)
        base_name = self.name_generator.get_name()
        name = f"{prefix} {base_name}"
        return self.breeder.limit_name_length(name)

    def generate_cat(self, parents: List[str] = None) -> Cat:
        is_female = random.choice([True, False])
        name = self.generate_name(is_female)
        color = random.choice(self.colors)
        pattern = random.choice(self.patterns)
        age = 0 if parents else random.randint(1, 15)
        personalities = self.generate_personalities(is_female)
        return Cat(name, color, pattern, personalities, age, is_female, parents)

    def generate_personalities(self, is_female):
        personalities = []
        for trait in self.personality_traits:
            value = random.randint(1, 10)
            if trait == "Shy" and is_female:
                value = min(10, value + 1)  # Females tend to be slightly shyer
            elif trait == "Energetic" and not is_female:
                value = min(10, value + 1)  # Males tend to be slightly more energetic
            personalities.append(Personality(trait, value))
        return personalities

    def evolve_names(self):
        self.name_generator.evolve()


class CatBreeder:
    def __init__(self, personality_change_rate: float = 0.1):
        self.current_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.cats: List[Cat] = []
        self.generation: int = 0
        self.year: int = 0
        self.cat_generator: CatGenerator = CatGenerator(self)
        self.personality_change_rate: float = personality_change_rate
        self.cat_food = 100  # 初始猫粮数量
        self.money = 5000  # 初始资金
        self.cat_food_consumption_rate = 10  # 每只猫每年消耗的猫粮量
        self.sick_probability = 0.1  # 10% chance for a cat to get sick each year
        self.treatment_cost = 100
        self.cat_food_cost = 2  # 每单位猫粮的成本
        self.food_shortage_months = 0  # 连续食物短缺的月数
        self.lifespan_limit_enabled = False
        self.cat_generation_cost = 500  # 生成一只猫的成本
        self.lottery_results = [
            (50, "什么都没有"),
            (30, "等价猫粮"),
            (15, "返还金额"),
            (4.9, "2倍金额"),
            (0.09, "10倍金额"),
            (0.001, "100倍金额")
        ]
        self.medal_level = 0  # 0: 无, 1: 铜, 2: 银, 3: 金
        self.gold_medals = 0
        self.cheat_mode = False

    def buy_medal(self):
        medal_prices = [10000, 50000, 100000]
        if self.cheat_mode:
            return False, "作弊模式下无法购买奖牌"
        if self.medal_level < 3:
            price = medal_prices[self.medal_level]
            if self.money >= price:
                self.money -= price
                self.medal_level += 1
                if self.medal_level == 3:
                    self.gold_medals = 1  # 首次达到金牌时，设置为1
                return True, f"成功购买{'铜' if self.medal_level == 1 else '银' if self.medal_level == 2 else '金'}奖牌"
            else:
                return False, "资金不足"
        else:
            if self.money >= 100000:
                self.money -= 100000
                self.gold_medals += 1
                return True, f"成功购买金奖牌，当前拥有 {self.gold_medals} 个金奖牌"
            else:
                return False, "资金不足"

    def cheat_money(self):
        self.cheat_mode = True
        self.money = float('inf')

    def play_lottery(self, amount):
        if self.money < amount:
            return "资金不足"

        self.money -= amount

        roll = random.random() * 100
        cumulative = 0
        for chance, prize in self.lottery_results:
            cumulative += chance
            if roll < cumulative:
                if prize == "什么都没有":
                    return "很遗憾，什么都没有中"
                elif prize == "等价猫粮":
                    self.cat_food += amount
                    return f"恭喜！获得 {amount} 单位猫粮"
                elif prize == "返还金额":
                    self.money += amount
                    return f"恭喜！返还 {amount} 元"
                elif prize == "2倍金额":
                    self.money += amount * 2
                    return f"恭喜！获得 {amount * 2} 元"
                elif prize == "10倍金额":
                    self.money += amount * 10
                    return f"太棒了！获得 {amount * 10} 元"
                elif prize == "100倍金额":
                    self.money += amount * 100
                    return f"难以置信！获得 {amount * 100} 元"

    def generate_cat(self):
        if self.money >= self.cat_generation_cost:
            new_cat = self.cat_generator.generate_cat()
            self.money -= self.cat_generation_cost
            self.add_cat(new_cat)
            return new_cat
        return None

    # def add_cat(self, cat):
    #     self.cats.append(cat)

    def set_lifespan_limit(self, enabled):
        self.lifespan_limit_enabled = enabled

    def advance_three_months(self):
        total_new_cats = []
        total_starved_cats = []
        total_old_cats = []
        total_shortage = 0

        for _ in range(3):
            self.current_date += datetime.timedelta(days=32)
            self.current_date = self.current_date.replace(day=1)
            new_cats, starved_cats, old_cats, shortage = self.advance_month()
            total_new_cats.extend(new_cats)
            total_starved_cats.extend(starved_cats)
            total_old_cats.extend(old_cats)
            total_shortage += shortage

        return total_new_cats, total_starved_cats, total_old_cats, total_shortage

    def heal_all_cats(self):
        sick_cats = [cat for cat in self.cats if cat.is_sick]
        total_cost = len(sick_cats) * self.treatment_cost

        if self.money >= total_cost:
            for cat in sick_cats:
                cat.heal()
            self.money -= total_cost
            return total_cost, len(sick_cats)
        else:
            affordable_cats = self.money // self.treatment_cost
            for cat in sick_cats[:affordable_cats]:
                cat.heal()
            self.money -= affordable_cats * self.treatment_cost
            return affordable_cats * self.treatment_cost, affordable_cats

    def treat_cat(self, cat):
        if self.money >= self.treatment_cost and cat.is_sick:
            cat.heal()
            self.money -= self.treatment_cost
            return True
        return False

    def calculate_cat_price(self, cat):
        base_price = 500
        age_factor = max(0, 1 - (cat.age / (20 * 12)))  # 假设猫的最大年龄是20年，现在用月份计算
        personality_factor = sum(p.value for p in cat.personalities) / (len(cat.personalities) * 10)
        gender_factor = 1.2 if cat.is_female else 1  # 假设雌性猫更值钱
        health_factor = 0.5 if cat.is_sick else 1  # 生病的猫价格减半

        price = int(base_price * age_factor * personality_factor * gender_factor * health_factor)
        return max(price, 50)  # 确保价格至少为50

    def handle_food_shortage(self, shortage):
        starved_cats = []
        if shortage > 0:
            shortage_percentage = shortage / (len(self.cats) * self.cat_food_consumption_rate)
            for cat in self.cats:
                if random.random() < shortage_percentage:
                    cat.get_sick()

            if self.food_shortage_months >= 2:
                starved_cats = self.remove_starving_cats()
        return starved_cats

    def remove_starving_cats(self):
        starving_cats = []
        for cat in self.cats[:]:  # 使用切片创建副本以便在循环中安全地移除元素
            if self.food_shortage_months >= 2 and random.random() < 0.3:  # 30% 概率死亡，只有在连续两个月短缺时才可能发生
                starving_cats.append(cat)
                self.cats.remove(cat)
        return starving_cats

    def advance_month(self, lifespan_limit=False):
        self.current_date += datetime.timedelta(days=32)
        self.current_date = self.current_date.replace(day=1)

        # 消耗猫粮
        shortage = self.consume_cat_food()

        # 处理食物短缺
        starved_cats = self.handle_food_shortage(shortage)

        # 猫咪年龄增长和性格变化
        for cat in self.cats:
            cat.update_age()
            cat.update_personality(self.personality_change_rate)

        # 处理疾病
        self.handle_diseases()

        # 移除过老的猫（如果启用了寿命限制）
        if lifespan_limit:
            old_cats = self.remove_old_cats()
        else:
            old_cats = []

        # 繁殖新的小猫
        new_cats = self.breed_cats()
        # starved_cats = self.remove_starving_cats()

        return new_cats, starved_cats, old_cats, shortage

    def consume_cat_food(self):
        required_food = len(self.cats) * self.cat_food_consumption_rate
        if self.cat_food >= required_food:
            self.cat_food -= required_food
            self.food_shortage_months = 0  # 重置食物短缺月数
            return 0  # 返回短缺量（这里是0）
        else:
            shortage = required_food - self.cat_food
            self.cat_food = 0
            self.food_shortage_months += 1
            return shortage  # 返回短缺量

    def buy_cat_food(self, amount: int):
        cost = amount * 2  # 假设每单位猫粮价格为 2
        if self.money >= cost:
            self.money -= cost
            self.cat_food += amount
            return True
        return False

    def sell_cat(self, cat: Cat) -> int:
        base_price = 500
        age_factor = max(0, 1 - (cat.age / 20))  # 年龄越大，价值越低
        personality_factor = sum(p.value for p in cat.personalities) / (len(cat.personalities) * 10)
        gender_factor = 1.2 if cat.is_female else 1  # 假设雌性猫更值钱

        price = int(base_price * age_factor * personality_factor * gender_factor)
        self.money += price
        self.cats.remove(cat)
        return price

    def handle_diseases(self):
        for cat in self.cats:
            if not cat.is_sick and random.random() < self.sick_probability:
                cat.get_sick()
            elif cat.is_sick:
                cat.try_recover()

    def advance_year(self):
        self.year += 1

        # 消耗猫粮
        self.consume_cat_food()

        # 猫咪年龄增长和性格变化
        for cat in self.cats:
            cat.age_up(self.personality_change_rate)

        # 处理疾病
        self.handle_diseases()

        # 移除过老的猫（如果启用了寿命限制）
        self.remove_old_cats()

        # 繁殖新的小猫
        new_cats = self.breed_cats()

        # 返回新出生的小猫列表
        return new_cats

    def remove_old_cats(self):
        if not self.lifespan_limit_enabled:
            return []
        old_cats = [cat for cat in self.cats if cat.age > random.randint(15 * 12, 25 * 12)]
        for cat in old_cats:
            self.cats.remove(cat)
        return old_cats

    def get_estimated_food_consumption(self):
        return len(self.cats) * self.cat_food_consumption_rate

    def add_cat(self, cat: Cat):
        self.cats.append(cat)

    def get_unique_name(self, name: str) -> str:
        existing_names = [cat.name for cat in self.cats]
        if name not in existing_names:
            return self.limit_name_length(name)

        count = 1
        while f"{name} ({count})" in existing_names:
            count += 1
        return self.limit_name_length(f"{name} ({count})")

    def limit_name_length(self, name: str) -> str:
        words = name.split()
        if len(words) <= 5:
            return name

        selected_words = [words[0]] + random.sample(words[1:], 4)
        return " ".join(selected_words)

    def breed_cats(self):
        new_cats = []
        eligible_cats = [cat for cat in self.cats if cat.can_breed()]

        for cat in eligible_cats:
            partner = self.find_partner(cat, eligible_cats)
            if partner:
                kitten = self.mate(cat, partner)
                if kitten:
                    new_cats.append(kitten)

        self.cats.extend(new_cats)
        return new_cats

    def find_partner(self, cat: Cat, eligible_cats: List[Cat]) -> Optional[Cat]:
        potential_partners = [c for c in eligible_cats if c.is_female != cat.is_female and c != cat]
        return random.choice(potential_partners) if potential_partners else None

    def mate(self, cat1: Cat, cat2: Cat) -> Optional[Cat]:
        if random.random() < 0.8:  # 80% chance of successful mating
            kitten_name = self.cat_generator.generate_name(random.choice([True, False]))
            kitten_name = self.get_unique_name(kitten_name)
            kitten_color = random.choice([cat1.color, cat2.color])
            kitten_pattern = random.choice([cat1.pattern, cat2.pattern])
            kitten_female = random.choice([True, False])
            kitten_personalities = self.cat_generator.generate_personalities(kitten_female)
            return Cat(kitten_name, kitten_color, kitten_pattern, kitten_personalities, 0, kitten_female,
                       [cat1.name, cat2.name])
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "current_date": self.current_date.isoformat(),
            "cats": [cat.to_dict() for cat in self.cats],
            "money": self.money,
            "cat_food": self.cat_food,
            "cat_food_consumption_rate": self.cat_food_consumption_rate,
            "food_shortage_months": self.food_shortage_months,
            "cat_food_cost": self.cat_food_cost,
            "personality_change_rate": self.personality_change_rate,
            "treatment_cost": self.treatment_cost,
            "medal_level": self.medal_level,
            "gold_medals": self.gold_medals,
            "cheat_mode": self.cheat_mode
        }
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.md5((data_str + SECRET_KEY).encode()).hexdigest()
        return {"data": data, "checksum": checksum}

    @classmethod
    def from_dict(cls, full_data: Dict[str, Any]) -> 'CatBreeder':
        try:
            data = full_data["data"]
            stored_checksum = full_data["checksum"]
            calculated_checksum = hashlib.md5((json.dumps(data, sort_keys=True) + SECRET_KEY).encode()).hexdigest()

            if stored_checksum != calculated_checksum:
                logger.warning("Data checksum mismatch detected")
                raise ValueError("数据已被修改，无法加载")

            breeder = cls()
            breeder.current_date = datetime.datetime.fromisoformat(data["current_date"])
            breeder.cats = [Cat.from_dict(cat_data) for cat_data in data["cats"]]
            breeder.money = data["money"]
            breeder.cat_food = data["cat_food"]
            breeder.cat_food_consumption_rate = data["cat_food_consumption_rate"]
            breeder.food_shortage_months = data["food_shortage_months"]
            breeder.cat_food_cost = data["cat_food_cost"]
            breeder.personality_change_rate = data["personality_change_rate"]
            breeder.treatment_cost = data["treatment_cost"]
            breeder.medal_level = data.get("medal_level", 0)
            breeder.gold_medals = data.get("gold_medals", 0)
            breeder.cheat_mode = data.get("cheat_mode", False)
            logger.info("CatBreeder instance successfully created from dictionary")
            return breeder
        except KeyError as e:
            logger.error(f"Missing key in data dictionary: {str(e)}", exc_info=True)
            raise ValueError(f"数据格式错误：缺少键 {str(e)}")
        except Exception as e:
            logger.error(f"Error creating CatBreeder instance from dictionary: {str(e)}", exc_info=True)
            raise


class CatBreederGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("小猫家园")
        self.breeder = CatBreeder()
        self.lifespan_enabled = tk.BooleanVar(value=False)  # 确保默认值为 False
        self.money_label = tk.StringVar(value=f"资金: {self.breeder.money}元")
        self.cat_food_label = tk.StringVar(value=f"猫粮: {self.breeder.cat_food}单位")
        self.calendar_label = tk.StringVar()
        self.estimated_consumption_label = tk.StringVar(value="预计月消耗: 0单位")
        self.gen_count = tk.StringVar(value="生成 0 个小猫")
        self.year_count = tk.StringVar(value=f"年份: {self.breeder.current_date.year}")  # 添加这行
        self.medal_photo = None
        self.medal_button_text = tk.StringVar(value="购买铜奖牌 (10000元)")
        self.medal_size = (40, 40)  # 设置默认大小
        self.load_medal_image()
        self.create_widgets()
        self.update_calendar()

    def load_medal_image(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # image_path = os.path.join(script_dir, "model.png")
            #
            # if not os.path.exists(image_path):
            #     print(f"警告：奖牌图像文件不存在：{image_path}")
            #     self.medal_photo = None
            #     return

            # self.original_medal_image = Image.open(image_path).convert("RGBA")
            self.original_medal_image = get_image()
            self.original_medal_image = self.original_medal_image.resize(self.medal_size, Image.LANCZOS)
            self.medal_photo = ImageTk.PhotoImage(self.original_medal_image)
        except Exception as e:
            print(f"加载奖牌图像时出错：{str(e)}")
            self.medal_photo = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(sticky=(tk.W, tk.E, tk.N, tk.S))

        left_frame = ttk.Frame(main_frame, padding="5", relief="sunken", borderwidth=1)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 5))

        right_frame = ttk.Frame(main_frame, padding="5", relief="sunken", borderwidth=1)
        right_frame.grid(row=0, column=1, sticky=(tk.E, tk.N, tk.S, tk.W))

        self.create_left_frame(left_frame)
        self.create_right_frame(right_frame)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def create_left_frame(self, parent):
        # 标题
        ttk.Label(parent, text="操作以及显示区域", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, pady=10)

        # 信息显示
        ttk.Label(parent, textvariable=self.year_count).grid(row=1, column=0, pady=5)
        ttk.Label(parent, textvariable=self.calendar_label).grid(row=2, column=0, pady=5)
        ttk.Label(parent, textvariable=self.money_label).grid(row=3, column=0, pady=5)
        ttk.Label(parent, textvariable=self.cat_food_label).grid(row=4, column=0, pady=5)
        ttk.Label(parent, textvariable=self.estimated_consumption_label).grid(row=5, column=0, pady=5)
        ttk.Label(parent, textvariable=self.gen_count).grid(row=6, column=0, pady=5)

        # 抽奖部分
        ttk.Label(parent, text="抽奖金额:").grid(row=7, column=0, pady=5)
        self.lottery_amount = tk.StringVar(value="10")
        ttk.Entry(parent, textvariable=self.lottery_amount).grid(row=8, column=0, pady=5)
        ttk.Button(parent, text="抽奖", command=self.play_lottery).grid(row=9, column=0, pady=5)

        # 主要操作按钮
        ttk.Button(parent, text="生成新小猫（500元）", command=self.generate_cats).grid(row=10, column=0, pady=5)
        ttk.Button(parent, text="时间前进（1个月）", command=self.advance_one_month).grid(row=11, column=0, pady=5)
        ttk.Button(parent, text="时间前进（3个月）", command=self.advance_three_months).grid(row=12, column=0, pady=5)

        # 购买猫粮
        ttk.Button(parent, text="购买猫粮（2元）", command=self.buy_cat_food).grid(row=13, column=0, pady=5)
        self.cat_food_amount = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.cat_food_amount).grid(row=14, column=0, pady=5)

        # 其他设置和操作
        ttk.Checkbutton(parent, text="启用寿命限制", variable=self.lifespan_enabled,
                        command=self.toggle_lifespan_limit).grid(row=15, column=0, pady=5)

        ttk.Label(parent, text="性格变化率:").grid(row=16, column=0, pady=5)
        self.personality_change_rate = tk.StringVar(value="0.1")
        ttk.Entry(parent, textvariable=self.personality_change_rate).grid(row=17, column=0, pady=5)

        ttk.Button(parent, text="治疗所有生病的猫", command=self.heal_all_cats).grid(row=18, column=0, pady=5)
        ttk.Button(parent, text="保存", command=self.save_state).grid(row=19, column=0, pady=5)
        ttk.Button(parent, text="读取", command=self.load_state).grid(row=20, column=0, pady=5)
        ttk.Button(parent, text="游戏说明", command=self.show_instructions).grid(row=21, column=0, pady=5)
        ttk.Button(parent, text="作弊模式", command=self.cheat_money).grid(row=22, column=0, pady=5)

        # 奖牌相关
        self.medal_count_label = ttk.Label(parent, text="")
        self.medal_count_label.grid(row=23, column=0, pady=5)
        self.buy_medal_button = ttk.Button(parent, textvariable=self.medal_button_text, command=self.buy_medal)
        self.buy_medal_button.grid(row=24, column=0, sticky="ew", pady=5)

        # 奖牌大小调节
        ttk.Label(parent, text="奖牌大小:").grid(row=25, column=0, sticky="w", pady=5)
        self.size_slider = ttk.Scale(parent, from_=20, to=100, orient=tk.HORIZONTAL, command=self.on_size_change)
        self.size_slider.set(40)  # 设置初始值
        self.size_slider.grid(row=26, column=0, sticky="ew", pady=5)

    def on_size_change(self, value):
        size = int(float(value))
        self.set_medal_size((size, size))

    def update_medal_button_text(self):
        medal_names = ["铜", "银", "金"]
        medal_prices = [10000, 50000, 100000]
        if self.breeder.medal_level < 3:
            next_medal = medal_names[self.breeder.medal_level]
            price = medal_prices[self.breeder.medal_level]
            new_text = f"购买{next_medal}奖牌 ({price}元)"
        else:
            new_text = f"购买金奖牌 (100000元) 现有{self.breeder.gold_medals}个"
        self.medal_button_text.set(new_text)
        # print(f"Updated button text to: {new_text}")  # 保留这行用于调试

    def update_medal_display(self):
        if not hasattr(self, 'original_medal_image') or self.original_medal_image is None:
            return

        colors = [(128, 128, 128), (205, 127, 50), (192, 192, 192), (255, 215, 0)]  # RGB values
        color = colors[self.breeder.medal_level]

        # 创建一个彩色图层
        color_layer = Image.new('RGBA', self.medal_size, color + (255,))

        # 将原始图像与彩色图层混合
        colored_image = ImageChops.multiply(self.original_medal_image, color_layer)

        self.medal_photo = ImageTk.PhotoImage(colored_image)
        if hasattr(self, 'medal_label'):
            self.medal_label.configure(image=self.medal_photo)

        if self.breeder.medal_level == 3:
            self.medal_count_label.configure(text=f"x{self.breeder.gold_medals}")
        else:
            self.medal_count_label.configure(text="")

        self.update_medal_button_text()
        # 在方法结束时添加这行
        self.master.update_idletasks()  # 强制更新UI

    def set_medal_size(self, size):
        """设置奖牌大小并重新加载图像"""
        self.medal_size = size
        self.load_medal_image()
        self.update_medal_display()

    def buy_medal(self):
        success, message = self.breeder.buy_medal()
        if success:
            messagebox.showinfo("购买成功", message)
            self.update_medal_display()
            self.update_economy_labels()
            self.update_medal_button_text()  # 确保这里调用更新按钮文字
        else:
            messagebox.showerror("购买失败", message)

    def play_lottery(self):
        try:
            amount = int(self.lottery_amount.get())
            if amount <= 0:
                messagebox.showerror("错误", "请输入正数金额")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            return
        result = self.breeder.play_lottery(amount)
        messagebox.showinfo("抽奖结果", result)
        self.update_economy_labels()

    def show_instructions(self):
        instruction_window = tk.Toplevel(self.master)
        instruction_window.title("游戏说明")
        instruction_window.geometry("600x400")

        instruction_text = scrolledtext.ScrolledText(instruction_window, wrap=tk.WORD, width=70, height=20)
        instruction_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        instructions = """
        猫咪繁殖器游戏说明：

        1. 基本玩法：
           - 生成新猫咪：每只新猫咪需要花费500元。
           - 时间推进：可以选择推进1个月或3个月，猫咪会随时间成长。
           - 猫粮管理：每只猫每月消耗一定量的猫粮，注意及时补充。

        2. 猫咪属性：
           - 猫咪有名字、性别、年龄、健康状态、性格等属性。
           - 猫咪到达一定年龄后可以繁殖。
           - 猫咪会生病，需要及时治疗。

        3. 经济系统：
           - 初始资金为5000元。
           - 可以通过出售猫咪赚取资金。
           - 需要购买猫粮和支付治疗费用。

        4. 寿命系统：
           - 可以选择是否启用寿命限制。
           - 启用后，猫咪会在15-25年后离世。

        5. 抽奖系统：
           - 可以投入一定金额参与抽奖。
           - 奖励包括：返还金额、获得猫粮、倍数奖励等。

        6. 游戏目标：
           - 繁衍更多的猫咪。
           - 保持猫咪健康。
           - 管理好经济，不要破产。
           - 尽可能的购买奖牌。

        祝您玩得开心！
        """

        instruction_text.insert(tk.END, instructions)
        instruction_text.config(state=tk.DISABLED)

    def toggle_lifespan_limit(self):
        self.breeder.set_lifespan_limit(self.lifespan_enabled.get())

    def advance_one_month(self):
        self._advance_time(1)

    def advance_three_months(self):
        self._advance_time(3)

    def _advance_time(self, months):
        try:
            self.breeder.personality_change_rate = float(self.personality_change_rate.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的性格变化率（浮点数）")
            return

        total_new_cats = []
        total_starved_cats = []
        total_old_cats = []
        total_shortage = 0

        for _ in range(months):
            # new_cats, starved_cats, old_cats, shortage = self.breeder.advance_month()
            new_cats, starved_cats, old_cats, shortage = self.breeder.advance_month(self.lifespan_enabled.get())
            total_new_cats.extend(new_cats)
            total_starved_cats.extend(starved_cats)
            total_old_cats.extend(old_cats)
            total_shortage += shortage

        self.update_calendar()
        self.update_cat_list()
        self.update_economy_labels()

        message = f"时间前进了{months}个月，现在是 {self.breeder.current_date.strftime('%Y年%m月')}。\n"

        if total_new_cats:
            message += f"这{months}个月内产生了 {len(total_new_cats)} 只新小猫。\n"
        else:
            message += f"这{months}个月内没有新的小猫出生。\n"

        if total_shortage > 0:
            message += f"这{months}个月内总共猫粮短缺 {total_shortage} 单位。\n"
            if self.breeder.food_shortage_months > 0:
                message += f"目前已经连续 {self.breeder.food_shortage_months} 个月猫粮不足。\n"

        death_message = ""
        if total_starved_cats:
            death_message += f"饥饿死亡: {len(total_starved_cats)} 只\n{', '.join([cat.name for cat in total_starved_cats])}\n\n"

        if total_old_cats:
            death_message += f"老年死亡: {len(total_old_cats)} 只\n{', '.join([cat.name for cat in total_old_cats])}\n"

        if death_message:
            messagebox.showinfo("猫咪离世", death_message)

        sick_cats_count = len([cat for cat in self.breeder.cats if cat.is_sick])
        message += f"目前有 {sick_cats_count} 只猫生病。"

        messagebox.showinfo("时间前进", message)

    def create_right_frame(self, parent):
        # 创建一个框架来容纳标题和奖牌
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(10, 5))
        header_frame.columnconfigure(1, weight=1)  # 让标题占据更多空间

        # 添加标题
        ttk.Label(header_frame, text="小猫园地", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w")

        # 创建一个框架来容纳奖牌和计数
        medal_frame = ttk.Frame(header_frame)
        medal_frame.grid(row=0, column=1, sticky="e")

        # 在medal_frame中添加奖牌显示
        if self.medal_photo:
            self.medal_label = ttk.Label(medal_frame, image=self.medal_photo)
            self.medal_label.pack(side=tk.LEFT, padx=(0, 5))
        else:
            self.medal_label = ttk.Label(medal_frame, text="奖牌")
            self.medal_label.pack(side=tk.LEFT, padx=(0, 5))

        self.medal_count_label = ttk.Label(medal_frame, text="")
        self.medal_count_label.pack(side=tk.LEFT)

        # 创建Treeview来显示猫咪列表
        self.cat_tree = ttk.Treeview(parent, columns=(
            '名字', '性别', '年龄', '健康状态', '价格', '性格', '叫声', '颜色', '亲代'),
                                     show='headings', selectmode='extended')

        # 设置列标题、宽度和排序功能
        for col in ('名字', '性别', '年龄', '健康状态', '价格', '性格', '叫声', '颜色', '亲代'):
            self.cat_tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(_col, False))
            self.cat_tree.column(col, width=100)  # 可以根据需要调整宽度

        self.cat_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.cat_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.cat_tree.configure(yscrollcommand=scrollbar.set)

        # 创建底部框架
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        bottom_frame.columnconfigure(1, weight=1)  # 让选择信息标签占据更多空间

        # 添加批量卖出按钮
        self.batch_sell_button = ttk.Button(bottom_frame, text="批量卖出", command=self.batch_sell_cats)
        self.batch_sell_button.grid(row=0, column=0, padx=(0, 10))

        # 添加选择信息标签
        self.selection_info = ttk.Label(bottom_frame, text="已选择: 0 只猫咪, 总价值: 0 元")
        self.selection_info.grid(row=0, column=1, sticky="w")

        # 绑定选择事件
        self.cat_tree.bind("<<TreeviewSelect>>", self.update_selection_info)
        self.cat_tree.bind("<Double-1>", self.on_cat_double_click)

        # 添加选择快捷键
        self.cat_tree.bind("<Control-a>", self.select_all_cats)
        self.cat_tree.bind("<Control-A>", self.select_all_cats)  # 大写A
        self.cat_tree.bind("<Control-d>", self.deselect_all_cats)
        self.cat_tree.bind("<Control-D>", self.deselect_all_cats)  # 大写D

        # 配置grid权重
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

    def update_selection_info(self, event=None):
        selected_items = self.cat_tree.selection()
        total_price = sum(self.breeder.calculate_cat_price(cat) for cat in self.breeder.cats
                          if cat.name in [self.cat_tree.item(item, "values")[0] for item in selected_items])
        self.selection_info.config(text=f"已选择: {len(selected_items)} 只猫咪, 总价值: {total_price} 元")

    def select_all_cats(self, event):
        for item in self.cat_tree.get_children():
            self.cat_tree.selection_add(item)

    def deselect_all_cats(self, event):
        self.cat_tree.selection_remove(self.cat_tree.get_children())

    def update_calendar(self):
        current_date = self.breeder.current_date
        self.calendar_label.set(current_date.strftime("%Y年%m月"))

    def treeview_sort_column(self, col, reverse):
        l = [(self.cat_tree.set(k, col), k) for k in self.cat_tree.get_children('')]
        l.sort(key=lambda t: self.sort_key(t[0]), reverse=reverse)

        # 重新排序
        for index, (val, k) in enumerate(l):
            self.cat_tree.move(k, '', index)

        # 反转排序顺序
        self.cat_tree.heading(col, command=lambda: self.treeview_sort_column(col, not reverse))

    def batch_sell_cats(self):
        selected_items = self.cat_tree.selection()
        if not selected_items:
            messagebox.showinfo("提示", "请先选择要卖出的猫咪")
            return

        cats_to_sell = []
        total_price = 0

        for item in selected_items:
            cat_name = self.cat_tree.item(item, "values")[0]
            cat = next((cat for cat in self.breeder.cats if cat.name == cat_name), None)
            if cat:
                price = self.breeder.calculate_cat_price(cat)
                total_price += price
                cats_to_sell.append(cat)

        if messagebox.askyesno("确认批量卖出",
                               f"确定要卖出选中的 {len(cats_to_sell)} 只猫咪吗？\n总价: {total_price} 元"):
            # 使用列表推导式更新 self.breeder.cats
            self.breeder.cats = [cat for cat in self.breeder.cats if cat not in cats_to_sell]
            self.breeder.money += total_price
            self.update_cat_list()
            self.update_economy_labels()
            messagebox.showinfo("批量卖出成功", f"成功卖出 {len(cats_to_sell)} 只猫咪，获得 {total_price} 元")

    def on_cat_double_click(self, event):
        selected_items = self.cat_tree.selection()
        if not selected_items:  # 如果没有选中任何项目
            return  # 直接返回，不执行任何操作

        item = selected_items[0]  # 获取选中的第一个项目
        cat_name = self.cat_tree.item(item, "values")[0]
        cat = next((cat for cat in self.breeder.cats if cat.name == cat_name), None)

        if not cat:  # 如果找不到对应的猫咪
            messagebox.showerror("错误", "无法找到选中的猫咪")
            return

        if cat.is_sick:
            if messagebox.askyesno("治疗猫咪", f"{cat_name} 生病了。要花费 {self.breeder.treatment_cost} 元治疗吗？"):
                if self.breeder.treat_cat(cat):
                    messagebox.showinfo("治疗成功", f"{cat_name} 已被治愈")
                else:
                    messagebox.showerror("治疗失败", "资金不足")
        else:
            price = self.breeder.calculate_cat_price(cat)
            if messagebox.askyesno("出售猫猫", f"确定要以 {price} 元的价格出售 {cat_name} 吗？"):
                self.breeder.sell_cat(cat)
                messagebox.showinfo("出售成功", f"{cat_name} 已被出售，获得 {price} 元")

        self.update_cat_list()
        self.update_economy_labels()

    def buy_cat_food(self):
        try:
            amount = int(self.cat_food_amount.get())
            if self.breeder.buy_cat_food(amount):
                messagebox.showinfo("购买成功", f"成功购买 {amount} 单位猫粮")
            else:
                messagebox.showerror("购买失败", "资金不足")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的猫粮数量")
        self.update_economy_labels()

    def heal_all_cats(self):
        cost, healed = self.breeder.heal_all_cats()
        if healed > 0:
            messagebox.showinfo("治疗成功", f"已治愈 {healed} 只生病的猫，花费 {cost} 元")
        else:
            messagebox.showinfo("治疗", "没有需要治疗的猫")
        self.update_cat_list()
        self.update_economy_labels()

    def cheat_money(self):
        self.breeder.cheat_money()
        self.update_economy_labels()
        self.buy_medal_button.configure(state="disabled")
        messagebox.showinfo("作弊模式", "已启用无限金钱，但无法购买奖牌")

    def update_economy_labels(self):
        self.money_label.set(f"资金: {self.breeder.money}元")
        self.cat_food_label.set(f"猫粮: {self.breeder.cat_food}单位")
        self.year_count.set(f"年份: {self.breeder.current_date.year}")  # 添加这行
        estimated_consumption = self.breeder.get_estimated_food_consumption()
        self.estimated_consumption_label.set(f"预计月消耗: {estimated_consumption}单位")
        self.gen_count.set(f"现有 {len(self.breeder.cats)} 个小猫")
        self.update_medal_button_text()  # 在这里也添加更新调用

    def generate_cats(self):
        new_cat = self.breeder.generate_cat()
        if new_cat:
            self.update_cat_list()
            self.update_economy_labels()
            messagebox.showinfo("新猫咪",
                                f"成功生成了一只新猫咪：{new_cat.name}，花费了 {self.breeder.cat_generation_cost} 元")
        else:
            messagebox.showerror("生成失败", f"资金不足，无法生成新猫咪。需要 {self.breeder.cat_generation_cost} 元")

    def breed_cats(self):
        new_cats = self.breeder.breed_cats()
        self.update_cat_list()
        messagebox.showinfo("繁殖结果", f"本次繁殖产生了 {len(new_cats)} 只新小猫")

    def advance_time(self):
        try:
            self.breeder.personality_change_rate = float(self.personality_change_rate.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的性格变化率（浮点数）")
            return

        new_cats, starved_cats, old_cats, shortage = self.breeder.advance_three_months()
        self.update_calendar()
        self.update_cat_list()
        self.update_economy_labels()

        message = f"时间前进了3个月，现在是 {self.breeder.current_date.strftime('%Y年%m月')}。\n"

        if new_cats:
            message += f"这3个月内产生了 {len(new_cats)} 只新小猫。\n"
        else:
            message += "这3个月内没有新的小猫出生。\n"

        if shortage > 0:
            message += f"这3个月内总共猫粮短缺 {shortage} 单位。\n"
            if self.breeder.food_shortage_months > 0:
                message += f"目前已经连续 {self.breeder.food_shortage_months} 个月猫粮不足。\n"

        if starved_cats:
            message += f"这3个月内共有 {len(starved_cats)} 只猫因饥饿死亡：{', '.join([cat.name for cat in starved_cats])}\n"

        if old_cats:
            message += f"这3个月内共有 {len(old_cats)} 只猫因年老死亡：{', '.join([cat.name for cat in old_cats])}\n"

        sick_cats_count = len([cat for cat in self.breeder.cats if cat.is_sick])
        message += f"目前有 {sick_cats_count} 只猫生病。"

        messagebox.showinfo("时间前进", message)

    def remove_old_cats(self):
        removal_messages = []
        cats_to_remove = []
        for cat in self.breeder.cats:
            if cat.age > random.randint(15, 25):
                removal_messages.append(f"小猫{cat.name}在地球呆了{cat.age}年，返回喵星")
                cats_to_remove.append(cat)

        for cat in cats_to_remove:
            self.breeder.cats.remove(cat)

        if removal_messages:
            messagebox.showinfo("喵星快报", "\n".join(removal_messages))

        self.update_cat_list()

    def sort_key(self, value):
        # 处理不同类型的值
        if value.endswith('元'):  # 价格
            return float(value[:-1])
        elif '岁' in value or '月' in value:  # 年龄
            return self.parse_age(value)
        elif value in ['雄性', '雌性']:  # 性别
            return value
        else:  # 其他情况，按字符串处理
            return value

    def parse_age(self, age_str):
        total_months = 0
        if '岁' in age_str:
            years, months = age_str.split('岁')
            total_months += int(years) * 12
            if '个月' in months:
                total_months += int(months.split('个月')[0])
        elif '个月' in age_str:
            total_months = int(age_str.split('个月')[0])
        return total_months

    def update_cat_list(self):
        self.cat_tree.delete(*self.cat_tree.get_children())
        for cat in self.breeder.cats:
            health_status = "生病" if cat.is_sick else "健康"
            price = self.breeder.calculate_cat_price(cat)
            personalities = ', '.join([f"{p.trait}({p.value:.1f})" for p in cat.personalities])
            self.cat_tree.insert('', 'end', values=(
                cat.name,
                "雌性" if cat.is_female else "雄性",
                cat.age_display(),
                health_status,
                f"{price}元",
                personalities,
                cat.meow(),
                f"{cat.color} {cat.pattern}",
                ", ".join(cat.parents) if cat.parents else "初代"
            ))

    def calculate_cat_price(self, cat: Cat) -> int:
        base_price = 500
        age_factor = max(0, 1 - (cat.age / 20))  # 年龄越大，价值越低
        personality_factor = sum(p.value for p in cat.personalities) / (len(cat.personalities) * 10)
        gender_factor = 1.2 if cat.is_female else 1  # 假设雌性猫更值钱
        health_factor = 0.5 if cat.is_sick else 1  # 生病的猫价格减半

        price = int(base_price * age_factor * personality_factor * gender_factor * health_factor)
        return price

    def load_state(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                self.breeder = CatBreeder.from_dict(full_data)
                self.update_cat_list()
                self.update_economy_labels()
                self.update_calendar()
                self.update_medal_display()
                messagebox.showinfo("读取成功", "已加载保存的状态")
                logger.info(f"State loaded successfully from {file_path}")
            except ValueError as e:
                error_msg = f"数据验证错误: {str(e)}"
                messagebox.showerror("错误", error_msg)
                logger.error(error_msg, exc_info=True)
            except json.JSONDecodeError:
                error_msg = "无法解析保存的数据"
                messagebox.showerror("错误", error_msg)
                logger.error(error_msg, exc_info=True)
            except Exception as e:
                error_msg = f"加载状态时发生错误: {str(e)}"
                messagebox.showerror("错误", error_msg)
                logger.error(error_msg, exc_info=True)

    def save_state(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                data = self.breeder.to_dict()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("保存成功", "当前状态已保存")
                logger.info(f"State saved successfully to {file_path}")
            except Exception as e:
                error_msg = f"保存状态时发生错误: {str(e)}"
                messagebox.showerror("保存失败", error_msg)
                logger.error(error_msg, exc_info=True)


if __name__ == "__main__":
    random.seed()  # 使用系统时间作为随机种子
    root = tk.Tk()
    app = CatBreederGUI(root)
    root.mainloop()
