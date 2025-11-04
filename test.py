def create_save_directory(base_dir="results"):
    import os
    import time
    """创建保存目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"direct_training_{timestamp}")      
    # 创建子目录
    print(f"创建目录: {save_dir}")
    
    return save_dir

if __name__ == "__main__":
    save_dir = create_save_directory()