import numpy as np
from tqdm import tqdm


class MRFSmoother:
    def __init__(
        self,
        num_classes,
        smoothness=1.0,
        iterations=10,
        edge_preserve_classes=None,
        min_region_size=6,
        preserve_small_classes=None,
        locked_classes=None,
    ):
        self.num_classes = num_classes
        self.smoothness = smoothness
        self.iterations = iterations
        self.edge_preserve_classes = set(edge_preserve_classes or [])
        self.min_region_size = min_region_size
        self.preserve_small_classes = set(preserve_small_classes or [])
        self.locked_classes = set(locked_classes or [])

    def _neighbor_bonus(self, grid, y, x):
        height, width = grid.shape
        bonus = np.zeros(self.num_classes, dtype=np.float64)
        current_class = grid[y, x]

        neighbors = []
        if y > 0:
            neighbors.append(grid[y - 1, x])
        if y < height - 1:
            neighbors.append(grid[y + 1, x])
        if x > 0:
            neighbors.append(grid[y, x - 1])
        if x < width - 1:
            neighbors.append(grid[y, x + 1])

        for neighbor_class in neighbors:
            weight = self.smoothness
            if current_class in self.edge_preserve_classes or neighbor_class in self.edge_preserve_classes:
                weight *= 0.5
            bonus[neighbor_class] += weight

        return bonus

    def _component_cells(self, grid, start_y, start_x, visited):
        target_class = grid[start_y, start_x]
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        cells = []

        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= grid.shape[0] or nx >= grid.shape[1]:
                    continue
                if visited[ny, nx] or grid[ny, nx] != target_class:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        return cells

    def _replacement_class(self, grid, cells):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        cell_set = set(cells)

        for y, x in cells:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= grid.shape[0] or nx >= grid.shape[1]:
                    continue
                if (ny, nx) in cell_set:
                    continue
                counts[grid[ny, nx]] += 1

        if counts.sum() == 0:
            return None
        return int(np.argmax(counts))

    def remove_small_regions(self, grid):
        if self.min_region_size <= 1:
            return grid

        cleaned = grid.copy()
        visited = np.zeros(grid.shape, dtype=bool)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if visited[y, x]:
                    continue

                cells = self._component_cells(cleaned, y, x, visited)
                component_class = cleaned[y, x]
                if component_class in self.preserve_small_classes:
                    continue
                if len(cells) >= self.min_region_size:
                    continue

                replacement = self._replacement_class(cleaned, cells)
                if replacement is None:
                    continue
                for cy, cx in cells:
                    cleaned[cy, cx] = replacement

        return cleaned

    def smooth(self, unary_probs):
        height, width, _ = unary_probs.shape
        grid = np.argmax(unary_probs, axis=-1)
        unary_scores = np.log(unary_probs + 1e-12)

        for _ in tqdm(range(self.iterations), desc="MRF smoothing"):
            changes = 0
            for y in range(height):
                for x in range(width):
                    if grid[y, x] in self.locked_classes:
                        continue
                    scores = unary_scores[y, x] + self._neighbor_bonus(grid, y, x)
                    new_class = int(np.argmax(scores))
                    if new_class != grid[y, x]:
                        grid[y, x] = new_class
                        changes += 1
            if changes == 0:
                break

        return self.remove_small_regions(grid)
