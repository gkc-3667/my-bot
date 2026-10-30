import os
import discord
from discord import app_commands
from discord.ext import commands
from github import Github
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")

# Khởi tạo bot với quyền mặc định
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Đã đồng bộ {len(synced)} lệnh slash.")
    except Exception as e:
        print(e)
    print(f"Bot đã đăng nhập thành công với tên {bot.user}")

@bot.tree.command(name="file", description="Tải file trực tiếp lên GitHub thông qua tài khoản của bạn")
@app_commands.describe(
    file="Chọn file code bạn muốn tải lên",
    path="Đường dẫn thư mục lưu trên GitHub (ví dụ: src/ hoặc để trống nếu ở thư mục gốc)"
)
@app_commands.checks.has_permissions(administrator=True) # Chỉ Quản trị viên server mới dùng được lệnh này
async def upload_file(interaction: discord.Interaction, file: discord.Attachment, path: str = ""):
    await interaction.response.defer(thinking=True)

    try:
        # Tải nội dung file từ Discord
        file_bytes = await file.read()
        
        # Xử lý đường dẫn thư mục lưu file
        file_path = f"{path.strip('/')}/{file.filename}" if path else file.filename

        # Kết nối tới GitHub bằng Token cá nhân
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)

        # Kiểm tra file đã tồn tại chưa để update hoặc tạo mới
        commit_message = f"Upload {file.filename} tự động qua lệnh Discord bot"
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, commit_message, file_bytes, contents.sha)
            action = "cập nhật thành công"
        except Exception:
            repo.create_file(file_path, commit_message, file_bytes)
            action = "tạo mới thành công"

        await interaction.followup.send(f"Thành công! Đã {action} file `{file.filename}` lên GitHub tại đường dẫn: `{file_path}`.")

    except Exception as e:
        await interaction.followup.send(f"Có lỗi xảy ra khi đẩy file lên GitHub: ```{e}```")

# Xử lý lỗi nếu người không phải Admin cố tình dùng lệnh
@upload_file.error
async def upload_file_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.errors.MissingPermissions):
        await interaction.response.send_message("Bạn không có quyền Quản trị viên (Administrator) để dùng lệnh này!", ephemeral=True)
    else:
        await interaction.response.send_message(f"Đã xảy ra lỗi: {error}", ephemeral=True)

# Chạy bot bằng token lấy từ file .env
bot.run(DISCORD_TOKEN)