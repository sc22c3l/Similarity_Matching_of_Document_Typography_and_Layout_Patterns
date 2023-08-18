def dfs(x, y, grid, visited, group, current_group):
    if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or visited[x][y] or grid[x][y] == 0:
        return
    
    visited[x][y] = True
    group[x][y] = current_group  # Assign the current group number
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8 directions
    
    for dx, dy in directions:
        dfs(x + dx, y + dy, grid, visited, group, current_group)

def separate_groups(grid):
    if not grid:
        return []
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    group = [[0 for _ in range(cols)] for _ in range(rows)]
    
    current_group = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                current_group += 1
                dfs(i, j, grid, visited, group, current_group)

    # Separate each group into its own grid
    separated = []
    for g in range(1, current_group + 1):

                
        new_grid = [[1 if cell == g else 0 for cell in row]  for row in group]
        new_grid1 =[]
        for row in new_grid:
            if 1 in row:
                new_grid1.append(row)
        
        separated.append(new_grid1)
    
    

    return separated

# Example usage
grid = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

result = separate_groups(grid)
for r in result:
    # for row in r:
    #     print(row)
    print(r)
    print("---")
