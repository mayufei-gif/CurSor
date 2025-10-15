class TestClass:
    def __init__(self):
        self.value = 42
    
    # 测试多行字符串
    MULTILINE_STRING = """这是一个
多行字符串
测试"""
    
    def test_method(self):
        """测试方法"""
        try:
            print(self.MULTILINE_STRING)
            return True
        except Exception as e:
            print(f"错误: {e}")
            return False

# 测试运行
if __name__ == "__main__":
    test = TestClass()
    test.test_method()