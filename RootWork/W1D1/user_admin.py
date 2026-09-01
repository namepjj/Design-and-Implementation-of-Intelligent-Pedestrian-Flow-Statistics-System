class User:
    """普通用户类，包含用户基本信息和问候方法"""

    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def describe_user(self):
        """打印用户的信息"""
        print(f"用户信息：姓 = {self.last_name}，名 = {self.first_name}")

    def greet_user(self):
        """打印个性化的问候语"""
        print(f"你好，{self.last_name}{self.first_name}，欢迎你！")


class Admin(User):
    """管理员类，继承自 User，拥有额外的权限列表"""

    def __init__(self, first_name, last_name):
        """初始化父类属性，并添加管理员特有的权限属性"""
        super().__init__(first_name, last_name)
        self.privileges = ["can add post", "can delete post", "can ban user"]

    def show_privileges(self):
        """显示管理员的所有权限"""
        print(f"管理员 {self.last_name}{self.first_name} 拥有以下权限：")
        for privilege in self.privileges:
            print(f"  - {privilege}")


if __name__ == "__main__":
    # 创建 Admin 实例，并调用它所有的方法
    admin = Admin("三", "张")

    admin.describe_user()   # 继承自 User
    admin.greet_user()      # 继承自 User
    admin.show_privileges() # Admin 特有方法
