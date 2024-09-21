# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/21 18:15
@Auth ： max
@File ：Advanced.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import json
from typing import List, Dict, Any, Optional


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
    def __init__(self, name: str, color: str, pattern: str, personalities: List[Personality], age: int, is_female: bool,
                 parents: List[str] = None):
        self.name = name
        self.color = color
        self.pattern = pattern
        self.personalities = personalities
        self.age = age
        self.is_female = is_female
        self.parents = parents or []

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

    def can_breed(self) -> bool:
        return 2 <= self.age <= 18

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
        self.cats: List[Cat] = []
        self.generation: int = 0
        self.year: int = 0
        self.cat_generator: CatGenerator = CatGenerator(self)
        self.personality_change_rate: float = personality_change_rate

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

    def breed_cats(self) -> List[Cat]:
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
        return {
            "cats": [cat.to_dict() for cat in self.cats],
            "generation": self.generation,
            "year": self.year
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CatBreeder':
        breeder = cls()
        breeder.cats = [Cat.from_dict(cat_data) for cat_data in data["cats"]]
        breeder.generation = data["generation"]
        breeder.year = data["year"]
        return breeder


class CatBreederGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("小猫繁殖器")
        self.breeder = CatBreeder()
        self.lifespan_enabled = tk.BooleanVar(value=False)
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self.master, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S))

        ttk.Button(control_frame, text="生成新小猫", command=self.generate_cats).grid(row=0, column=0, pady=5)
        ttk.Button(control_frame, text="繁殖小猫", command=self.breed_cats).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="时间前进", command=self.advance_time).grid(row=2, column=0, pady=5)
        ttk.Button(control_frame, text="保存", command=self.save_state).grid(row=3, column=0, pady=5)
        ttk.Button(control_frame, text="读取", command=self.load_state).grid(row=4, column=0, pady=5)

        self.gen_count = tk.StringVar(value="生成 0 个小猫")
        ttk.Label(control_frame, textvariable=self.gen_count).grid(row=5, column=0, pady=5)

        self.year_count = tk.StringVar(value="年份: 0")
        ttk.Label(control_frame, textvariable=self.year_count).grid(row=6, column=0, pady=5)

        # 添加性格变化率输入框
        ttk.Label(control_frame, text="性格变化率:").grid(row=7, column=0, pady=5)
        self.personality_change_rate = tk.StringVar(value="0.1")
        ttk.Entry(control_frame, textvariable=self.personality_change_rate).grid(row=8, column=0, pady=5)

        cat_frame = ttk.Frame(self.master, padding="10")
        cat_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.cat_tree = ttk.Treeview(cat_frame, columns=('名字', '性别', '性格', '年龄', '叫声', '亲代'),
                                     show='headings')
        self.cat_tree.heading('名字', text='名字')
        self.cat_tree.heading('性别', text='性别')
        self.cat_tree.heading('性格', text='性格')
        self.cat_tree.heading('年龄', text='年龄')
        self.cat_tree.heading('叫声', text='叫声')
        self.cat_tree.heading('亲代', text='亲代')
        self.cat_tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        scrollbar = ttk.Scrollbar(cat_frame, orient=tk.VERTICAL, command=self.cat_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.cat_tree.configure(yscrollcommand=scrollbar.set)

        cat_frame.columnconfigure(0, weight=1)
        cat_frame.rowconfigure(0, weight=1)

        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)
        # Add lifespan checkbox
        ttk.Checkbutton(control_frame, text="启用寿命限制", variable=self.lifespan_enabled).grid(row=9, column=0,
                                                                                                 pady=5)

    def generate_cats(self):
        new_cat = self.breeder.cat_generator.generate_cat()
        self.breeder.add_cat(new_cat)
        self.update_cat_list()
        self.gen_count.set(f"生成 {len(self.breeder.cats)} 个小猫")

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

        if self.lifespan_enabled.get():
            self.remove_old_cats()

        new_cats = self.breeder.breed_cats()
        for cat in self.breeder.cats:
            cat.age_up(self.breeder.personality_change_rate)

        self.breeder.year += 1
        self.breeder.cat_generator.name_generator.evolve()
        self.update_cat_list()
        self.year_count.set(f"年份: {self.breeder.year}")
        messagebox.showinfo("时间前进", f"时间前进了1年，产生了 {len(new_cats)} 只新小猫")

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

    def update_cat_list(self):
        self.cat_tree.delete(*self.cat_tree.get_children())
        for cat in self.breeder.cats:
            personalities = ', '.join([f"{p.trait}({p.value:.2f})" for p in cat.personalities])
            self.cat_tree.insert('', 'end', values=(
                cat.name,
                "雌性" if cat.is_female else "雄性",
                personalities,
                f"{cat.age}岁",
                cat.meow(),
                ", ".join(cat.parents) if cat.parents else "初代"
            ))

    def save_state(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.breeder.to_dict(), f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", "当前状态已保存")

    def load_state(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.breeder = CatBreeder.from_dict(data)
            self.update_cat_list()
            self.gen_count.set(f"生成 {len(self.breeder.cats)} 个小猫")
            self.year_count.set(f"年份: {self.breeder.year}")
            messagebox.showinfo("读取成功", "已加载保存的状态")


def update_cat_list(self):
    self.cat_tree.delete(*self.cat_tree.get_children())
    for cat in self.breeder.cats:
        personalities = ', '.join([f"{p.trait}({p.value})" for p in cat.personalities])
        self.cat_tree.insert('', 'end', values=(
            cat.name,
            "雌性" if cat.is_female else "雄性",
            personalities,
            f"{cat.age}岁",
            cat.meow(),
            ", ".join(cat.parents) if cat.parents else "初代"
        ))

    def save_state(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.breeder.to_dict(), f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", "当前状态已保存")

    def load_state(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.breeder = CatBreeder.from_dict(data)
            self.update_cat_list()
            self.gen_count.set(f"生成 {len(self.breeder.cats)} 个小猫")
            self.year_count.set(f"年份: {self.breeder.year}")
            messagebox.showinfo("读取成功", "已加载保存的状态")


if __name__ == "__main__":
    random.seed()  # 使用系统时间作为随机种子
    root = tk.Tk()
    app = CatBreederGUI(root)
    root.mainloop()
