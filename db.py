import sqlite3
import logging

DB_PATH = "user.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                gender TEXT NOT NULL,
                phone TEXT NOT NULL,
                emergency_contact TEXT NOT NULL,
                password BLOB NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}", exc_info=True)

def is_valid_bcrypt_hash(pw_hash):
    if not isinstance(pw_hash, (bytes, str)):
        return False
    if isinstance(pw_hash, str):
        pw_hash = pw_hash.encode('utf-8')
    return pw_hash.startswith(b"$2a$") or pw_hash.startswith(b"$2b$") or pw_hash.startswith(b"$2y$")

def register_user(full_name, email, gender, phone, emergency_contact, password_hash):
    try:
        # Ensure password_hash is bytes before storing
        if isinstance(password_hash, str):
            password_hash = password_hash.encode('utf-8')
        if not is_valid_bcrypt_hash(password_hash):
            logging.error(f"Attempted to register user {email} with invalid bcrypt hash.")
            raise ValueError("Invalid bcrypt hash for password.")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (full_name, email, gender, phone, emergency_contact, password)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, email, gender, phone, emergency_contact, password_hash))
        conn.commit()
        conn.close()
        logging.info(f"User {email} registered successfully.")
    except sqlite3.IntegrityError:
        logging.warning(f"Attempt to register duplicate email: {email}")
        raise
    except Exception as e:
        logging.error(f"Error registering user {email}: {e}", exc_info=True)
        raise

def get_user_by_email(email):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            user_dict = dict(row)
            # Ensure password is bytes
            password = user_dict.get('password')
            if isinstance(password, str):
                try:
                    password = password.encode('utf-8')
                except Exception as e:
                    logging.error(f"Error encoding password from DB: {e}", exc_info=True)
            user_dict['password'] = password
            return user_dict
        return None
    except Exception as e:
        logging.error(f"Error fetching user by email {email}: {e}", exc_info=True)
        return None

def get_emergency_contact_by_email(email):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT emergency_contact FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logging.error(f"Error fetching emergency contact for {email}: {e}", exc_info=True)
        return None
