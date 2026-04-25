import boto3, os, mimetypes, json, sys
from botocore.config import Config

ACCOUNT_ID = "79b6f1ae5a01b8b8f23069e3b3af234e"
BUCKET     = "editions"
ACCESS_KEY_ID  = os.getenv("R2_ACCESS_KEY_ID")   # "Access Key ID" from R2 token page
SECRET_KEY     = os.getenv("R2_SECRET_KEY")       # "Secret Access Key" from R2 token page

EXCLUDED_FOLDERS = {"chunk_checkpoints"}
EXCLUDED_FILES   = {"cover_raw.png"} 
INDEX_CACHE = os.path.join(os.path.dirname(__file__), ".books_index.json")

s3 = boto3.client("s3",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto"
)

def load_index():
    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE) as f:
            return json.load(f).get("books", [])
    return []

def save_index(folders):
    with open(INDEX_CACHE, "w") as f:
        json.dump({"books": sorted(folders)}, f, indent=2)

def upload_folder(local_path):
    folder_name = os.path.basename(local_path.rstrip("/\\"))
    print(f"\n📤 Upload de '{folder_name}'...")
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDERS]
        for filename in files:
            if filename in EXCLUDED_FILES:
                continue
            local_file   = os.path.join(root, filename)
            relative     = os.path.relpath(local_file, local_path)
            r2_key       = f"{folder_name}/{relative}".replace("\\", "/")
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            s3.upload_file(local_file, BUCKET, r2_key,
                ExtraArgs={"ContentType": content_type})
            print(f"  ✅ {r2_key}")
    return folder_name

def update_index(new_folder):
    books = load_index()
    if new_folder not in books:
        books.append(new_folder)
    books = sorted(books)
    save_index(books)

    index_bytes = json.dumps({"books": books}).encode()
    s3.put_object(Bucket=BUCKET, Key="index.json",
        Body=index_bytes, ContentType="application/json")
    print(f"\n📖 index.json mis à jour — {len(books)} livre(s) : {books}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python upload.py /chemin/vers/dossier_livre")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isdir(path):
        print(f"❌ Dossier introuvable : {path}")
        sys.exit(1)

    if not ACCESS_KEY_ID or not SECRET_KEY:
        print("❌ R2_ACCESS_KEY_ID ou R2_SECRET_KEY manquant")
        sys.exit(1)

    folder_name = upload_folder(path)
    update_index(folder_name)
    print("\n🐝 Terminé !")