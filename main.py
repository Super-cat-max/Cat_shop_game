# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import random
from typing import List, Optional


class Personality:
    """表示猫的性格特质"""

    def __init__(self, trait: str, value: int):
        self.trait = trait  # 特质名称
        self.value = value  # 特质强度 (1-10)


class Cat:
    """表示一只猫"""

    def __init__(self, name: str, color: str, pattern: str, personalities: List[Personality], age: int,
                 is_female: bool):
        self.name = name
        self.color = color
        self.pattern = pattern
        self.personalities = personalities
        self.age = age
        self.is_female = is_female

    def meow(self):
        """根据猫的性格发出喵喵声"""
        print(f"{self.name} says: ", end="")
        loudness = sum(p.value for p in self.personalities if p.trait == "Energetic")
        shyness = sum(p.value for p in self.personalities if p.trait == "Shy")

        if shyness > loudness:
            print("^..^ (soft meow)")
        elif loudness > shyness:
            print("MEOW! MEOW!")
        else:
            print("Meow~")

    def display_info(self):
        """显示猫的所有信息"""
        print(f"Name: {self.name}, Color: {self.color}, Pattern: {self.pattern}")
        print(f"Age: {self.age} years old, Gender: {'Female' if self.is_female else 'Male'}")
        print("Personalities: ", end="")
        for p in self.personalities:
            print(f"{p.trait}({p.value}) ", end="")
        print()

    def mate(self, partner: 'Cat') -> Optional['Cat']:
        """与另一只猫繁殖，返回一个小猫"""
        if self.is_female == partner.is_female:
            print("Cats must be of opposite genders to mate.")
            return None

        # 随机选择小猫的特征
        kitten_name = f"Kitten{random.randint(0, 999)}"
        kitten_color = random.choice([self.color, partner.color])
        kitten_pattern = random.choice([self.pattern, partner.pattern])
        kitten_female = random.choice([True, False])

        # 继承父母的性格，带有小的随机变异
        kitten_personalities = []
        for p1 in self.personalities:
            for p2 in partner.personalities:
                if p1.trait == p2.trait:
                    avg_value = (p1.value + p2.value) // 2
                    new_value = max(1, min(10, avg_value + random.randint(-1, 1)))
                    kitten_personalities.append(Personality(p1.trait, new_value))

        return Cat(kitten_name, kitten_color, kitten_pattern, kitten_personalities, 0, kitten_female)

    def can_breed(self) -> bool:
        """检查猫是否在可繁殖年龄 (1-10岁)"""
        return 1 <= self.age <= 10

    def age_up(self):
        """使猫长大一岁"""
        self.age += 1


class CatGenerator:
    """生成随机的猫"""

    def __init__(self):
        self.names = ["Whiskers", "Mittens", "Fluffy", "Oreo", "Luna", "Tiger", "Smokey", "Mochi", "Neko"]
        self.colors = ["Black", "White", "Orange", "Gray", "Brown"]
        self.patterns = ["Solid", "Tabby", "Calico", "Tortoiseshell", "Tuxedo", "Spotted"]
        self.personality_traits = ["Shy", "Energetic", "Lazy", "Curious", "Friendly"]

    def generate_cat(self) -> Cat:
        """生成一只随机的猫"""
        name = random.choice(self.names)
        color = random.choice(self.colors)
        pattern = random.choice(self.patterns)
        age = random.randint(1, 15)
        is_female = random.choice([True, False])

        personalities = [Personality(trait, random.randint(1, 10)) for trait in self.personality_traits]

        return Cat(name, color, pattern, personalities, age, is_female)


def main():
    generator = CatGenerator()
    cats = []

    # 生成初始的成年猫群
    num_cats = int(input("How many adult cats would you like to generate? "))
    for _ in range(num_cats):
        cats.append(generator.generate_cat())

    print("\nGenerated Adult Cats:")
    for cat in cats:
        cat.display_info()
        cat.meow()
        print()

    # 模拟多代繁殖
    num_generations = int(input("How many generations would you like to simulate? "))
    for gen in range(1, num_generations + 1):
        print(f"\nGeneration {gen}:")
        new_cats = []

        # 尝试让每对猫繁殖
        for i in range(0, len(cats), 2):
            if i + 1 < len(cats) and cats[i].can_breed() and cats[i + 1].can_breed():
                kitten = cats[i].mate(cats[i + 1])
                if kitten:
                    new_cats.append(kitten)
                    print("New kitten born:")
                    kitten.display_info()
                    kitten.meow()
                    print()

        # 所有猫长大一岁
        for cat in cats:
            cat.age_up()

        # 将新生的小猫加入到猫群中
        cats.extend(new_cats)

    print("\nFinal cat population:")
    for cat in cats:
        cat.display_info()
        cat.meow()
        print()


if __name__ == "__main__":
    main()