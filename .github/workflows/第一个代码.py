def init_population(pop_size, chrom_length):
    """
    初始化种群
    参数：pop_size-种群规模，chrom_length-染色体长度（如32）
    返回：种群列表（元素为字典，含"chromosome"和"fitness"）
    """
    population = []
    for _ in range(pop_size):
        # 随机生成chrom_length位二进制字符串（0/1）
        chromosome = ''.join([str(random.randint(0, 1)) for _ in range(chrom_length)])
        population.append({"chromosome": chromosome, "fitness": 0})  # 适应度初始化为0，后续计算
    return population