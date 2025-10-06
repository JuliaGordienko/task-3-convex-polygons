from __future__ import annotations
import math
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """Точка на плоскости"""
    x: float
    y: float

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def distance_to(self, other: Point) -> float:
        """Расстояние до другой точки"""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


class Shape(ABC):
    """Абстрактный базовый класс для фигур"""

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    @abstractmethod
    def contains(self, point: Point) -> bool:
        pass


class ConvexPolygon(Shape):
    """Класс выпуклых многоугольников"""

    def __init__(self, vertices: List[Tuple[float, float]]):
        """Инициализация с проверкой выпуклости"""
        if len(vertices) < 3:
            raise ValueError("Многоугольник должен иметь хотя бы 3 вершины")

        self._vertices = [Point(x, y) for x, y in vertices]

        if not self._is_convex():
            raise ValueError("Многоугольник не является выпуклым")

        self._order_vertices_ccw()

    def _cross_product(self, a: Point, b: Point, c: Point) -> float:
        """Векторное произведение векторов AB и AC"""
        ab = b - a
        ac = c - a
        return ab.x * ac.y - ab.y * ac.x

    def _is_convex(self) -> bool:
        """Проверка выпуклости с использованием векторных произведений"""
        n = len(self._vertices)
        positive = negative = False

        for i in range(n):
            a = self._vertices[i]
            b = self._vertices[(i + 1) % n]
            c = self._vertices[(i + 2) % n]

            cross = self._cross_product(a, b, c)

            if math.isclose(cross, 0):
                continue

            if cross > 0:
                positive = True
            else:
                negative = True

            if positive and negative:
                return False

        return True

    def _order_vertices_ccw(self):
        """Упорядочивание вершин против часовой стрелки"""
        if len(self._vertices) < 3:
            return

        start_idx = 0
        for i in range(1, len(self._vertices)):
            if (self._vertices[i].x < self._vertices[start_idx].x or
                    (math.isclose(self._vertices[i].x, self._vertices[start_idx].x) and
                     self._vertices[i].y < self._vertices[start_idx].y)):
                start_idx = i

        self._vertices = self._vertices[start_idx:] + self._vertices[:start_idx]

        if len(self._vertices) >= 3:
            cross = self._cross_product(self._vertices[0], self._vertices[1], self._vertices[2])
            if cross < 0:
                self._vertices = [self._vertices[0]] + self._vertices[1:][::-1]

    @property
    def vertices(self) -> List[Point]:
        """Свойство только для чтения доступа к вершинам"""
        return self._vertices.copy()

    @property
    def num_vertices(self) -> int:
        """Количество вершин"""
        return len(self._vertices)

    def area(self) -> float:
        """Вычисление площади методом векторного произведения (есть такакя Эформула шнуровки")"""
        n = len(self._vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self._vertices[i].x * self._vertices[j].y
            area -= self._vertices[j].x * self._vertices[i].y

        return abs(area) / 2.0

    def perimeter(self) -> float:
        """Вычисление периметра многоугольника"""
        n = len(self._vertices)
        perimeter = 0.0

        for i in range(n):
            j = (i + 1) % n
            perimeter += self._vertices[i].distance_to(self._vertices[j])

        return perimeter

    def contains(self, point: Union[Point, Tuple[float, float]]) -> bool:
        """Проверка нахождения точки внутри многоугольника"""
        if isinstance(point, tuple):
            point = Point(point[0], point[1])

        n = len(self._vertices)
        total_angle = 0.0

        for i in range(n):
            a = self._vertices[i] - point
            b = self._vertices[(i + 1) % n] - point

            dot = a.x * b.x + a.y * b.y
            cross = a.x * b.y - a.y * b.x
            angle = math.atan2(cross, dot)
            total_angle += angle

        return math.isclose(abs(total_angle), 2 * math.pi, abs_tol=1e-10)

    def intersection(self, other: ConvexPolygon) -> Optional[ConvexPolygon]:
        """Пересечение двух выпуклых многоугольников"""
        intersection_points = []

        # Вершины первого многоугольника внутри второго
        for vertex in self._vertices:
            if other.contains(vertex):
                intersection_points.append((vertex.x, vertex.y))

        # Вершины второго многоугольника внутри первого
        for vertex in other.vertices:
            if self.contains(vertex):
                intersection_points.append((vertex.x, vertex.y))

        # Точки пересечения рёбер
        for i in range(len(self._vertices)):
            a1 = self._vertices[i]
            a2 = self._vertices[(i + 1) % len(self._vertices)]

            for j in range(len(other.vertices)):
                b1 = other.vertices[j]
                b2 = other.vertices[(j + 1) % len(other.vertices)]

                intersection = self._find_line_intersection(a1, a2, b1, b2)
                if intersection:
                    intersection_points.append((intersection.x, intersection.y))

        # Удалить дубликаты
        unique_points = self._remove_duplicate_points(intersection_points)

        if len(unique_points) < 3:
            return None

        try:
            return ConvexPolygon(unique_points)
        except ValueError:
            return None

    def _find_line_intersection(self, a1: Point, a2: Point, b1: Point, b2: Point) -> Optional[Point]:
        """Находит точку пересечения двух отрезков"""
        da = Point(a2.x - a1.x, a2.y - a1.y)
        db = Point(b2.x - b1.x, b2.y - b1.y)

        det = da.x * db.y - da.y * db.x

        if math.isclose(det, 0):
            return None

        t = ((b1.x - a1.x) * db.y - (b1.y - a1.y) * db.x) / det
        u = ((b1.x - a1.x) * da.y - (b1.y - a1.y) * da.x) / det

        if 0 <= t <= 1 and 0 <= u <= 1:
            return Point(a1.x + t * da.x, a1.y + t * da.y)

        return None

    def _remove_duplicate_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Удаляет дубликаты точек"""
        unique = []
        for point in points:
            point_obj = Point(point[0], point[1])
            if not any(math.isclose(point_obj.x, p.x) and math.isclose(point_obj.y, p.y)
                       for p in [Point(*p) for p in unique]):
                unique.append(point)
        return unique

    def triangulate(self) -> List[List[Tuple[float, float]]]:
        """Триангуляция выпуклого многоугольника "методом веера" """
        n = len(self._vertices)
        triangles = []

        for i in range(1, n - 1):
            triangle = [
                (self._vertices[0].x, self._vertices[0].y),
                (self._vertices[i].x, self._vertices[i].y),
                (self._vertices[i + 1].x, self._vertices[i + 1].y)
            ]
            triangles.append(triangle)

        return triangles

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Возвращает прямоугольник который ограничивает (min_x, min_y, max_x, max_y)"""
        if not self._vertices:
            return 0, 0, 0, 0

        min_x = min(v.x for v in self._vertices)
        max_x = max(v.x for v in self._vertices)
        min_y = min(v.y for v in self._vertices)
        max_y = max(v.y for v in self._vertices)

        return min_x, min_y, max_x, max_y

    def __repr__(self) -> str:
        return f"ConvexPolygon({[(v.x, v.y) for v in self._vertices]})"


def demonstrate_comprehensive():
    """Полная демонстрация всех возможностей"""

    print(" ПОЛНАЯ ДЕМОНСТРАЦИЯ КЛАССА ConvexPolygon")
    print("=" * 50)

    # 1. Создание различных выпуклых многоугольников
    print("\n1.  СОЗДАНИЕ МНОГОУГОЛЬНИКОВ")
    print("-" * 30)

    shapes = {
        "Квадрат": ConvexPolygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        "Треугольник": ConvexPolygon([(1, 1), (3, 1), (2, 3)]),
        "Пятиугольник": ConvexPolygon([(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]),
        "Прямоугольник": ConvexPolygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
    }

    for name, shape in shapes.items():
        print(f"   {name}: {shape}")
        print(f"   Вершин: {shape.num_vertices}")

    # 2. Площадь и периметр
    print("\n2.  ПЛОЩАДЬ И ПЕРИМЕТР")
    print("-" * 30)

    for name, shape in shapes.items():
        print(f"   {name}:")
        print(f"     Площадь = {shape.area():.2f}")
        print(f"     Периметр = {shape.perimeter():.2f}")

    # 3. Принадлежность точек
    print("\n3.  ПРОВЕРКА ПРИНАДЛЕЖНОСТИ ТОЧЕК")
    print("-" * 30)

    test_points = [
        (1, 1), (3, 3), (0.5, 0.5), (2, 1),
        (1.5, 1.5), (5, 5), (0, 0)
    ]

    for point in test_points:
        print(f"   Точка {point}:")
        for name, shape in shapes.items():
            if shape.contains(point):
                print(f"      внутри {name}")

    # 4. Триангуляция
    print("\n4. ТРИАНГУЛЯЦИЯ")
    print("-" * 30)

    for name, shape in shapes.items():
        if shape.num_vertices > 3:
            triangles = shape.triangulate()
            print(f"   {name} разбит на {len(triangles)} треугольника:")
            for i, triangle in enumerate(triangles, 1):
                print(f"     Треугольник {i}: {triangle}")

    # 5. Пересечения
    print("\n5.  ПЕРЕСЕЧЕНИЯ МНОГОУГОЛЬНИКОВ")
    print("-" * 30)

    square = shapes["Квадрат"]
    triangle = shapes["Треугольник"]
    rectangle = shapes["Прямоугольник"]

    intersections = [
        ("Квадрат и Треугольник", square, triangle),
        ("Квадрат и Прямоугольник", square, rectangle),
        ("Треугольник и Прямоугольник", triangle, rectangle),
    ]

    for desc, poly1, poly2 in intersections:
        intersection = poly1.intersection(poly2)
        if intersection:
            print(f"   {desc}:")
            print(f"     Пересечение: {intersection}")
            print(f"     Площадь пересечения: {intersection.area():.2f}")
        else:
            print(f"   {desc}: пересечение пустое")

    # 6. Ограничивающие прямоугольники
    print("\n6. ОГРАНИЧИВАЮЩИЕ ПРЯМОУГОЛЬНИКИ")
    print("-" * 30)

    for name, shape in shapes.items():
        bbox = shape.get_bounding_box()
        print(f"   {name}: {bbox}")

    # 7. Проверка выпуклости
    print("\n7.  ПРОВЕРКА ВЫПУКЛОСТИ")
    print("-" * 30)

    test_cases = [
        ("Выпуклый", [(0, 0), (2, 0), (2, 2), (0, 2)]),
        ("Невыпуклый", [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)]),
        ("Треугольник", [(0, 0), (2, 0), (1, 2)]),
    ]

    for desc, vertices in test_cases:
        try:
            poly = ConvexPolygon(vertices)
            print(f"   {desc}: Успешно создан")
        except ValueError as e:
            print(f"   {desc}:  {e}")

    # 8. Сравнение многоугольников
    print("\n8. СРАВНЕНИЕ МНОГОУГОЛЬНИКОВ")
    print("-" * 30)

    square1 = ConvexPolygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    square2 = ConvexPolygon([(0, 0), (0, 2), (2, 2), (2, 0)])  # Тот же квадрат
    square3 = ConvexPolygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # Другой квадрат

    print(f"   Квадрат 1 == Квадрат 2: {square1 == square2}")
    print(f"   Квадрат 1 == Квадрат 3: {square1 == square3}")


def demonstrate_advanced_features():

    print("\n" + " Крутой многоугольник")
    print("=" * 50)

    # Создание крутого выпуклого многоугольника
    complex_poly = ConvexPolygon([
        (0, 0), (3, 0), (4, 1), (4, 3),
        (3, 4), (1, 4), (0, 3)
    ])

    print(f"Сложный многоугольник: {complex_poly}")
    print(f"Площадь: {complex_poly.area():.2f}")
    print(f"Периметр: {complex_poly.perimeter():.2f}")

    # Тест граничных случаев
    print("\n ГРАНИЧНЫЕ СЛУЧАИ:")
    edge_points = [(0, 0), (2, 0), (4, 2), (2, 4), (0, 2)]
    for point in edge_points:
        contains = complex_poly.contains(point)
        on_edge = " (на границе)" if contains else ""
        print(f"  Точка {point}: внутри{on_edge}")


if __name__ == "__main__":
    demonstrate_comprehensive()
    demonstrate_advanced_features()


