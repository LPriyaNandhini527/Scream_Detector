import sqlite3
import bcrypt
import logging

DB_PATH = "user.db"

def reset_user_passwords():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Fetch all users
        cursor.execute("SELECT id, password FROM users")
        users = cursor.fetchall()

        for user_id, pw in users:
            # Check if password is a valid bcrypt hash
            if not (isinstance(pw, bytes) and (pw.startswith(b"$2a$") or pw.startswith(b"$2b$") or pw.startswith(b"$2y$"))):
                # If invalid, reset password to a default secure password hash
                default_password = "ChangeMe123"
                hashed = bcrypt.hashpw(default_password.encode(), bcrypt.gensalt())
                cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, user_id))
                logging.info(f"Reset password for user id {user_id} to default password.")

        conn.commit()
        conn.close()
        print("User passwords reset completed. Default password is 'ChangeMe123' for affected users.")
    except Exception as e:
        logging.error(f"Error resetting user passwords: {e}", exc_info=True)

if __name__ == "__main__":
    reset_user_passwords()
