  
import socket
try:
    print(socket.getaddrinfo("db.arqxodzerbmbjnwiutes.supabase.co", 5432))
except Exception as e:
    print("getaddrinfo error:", e)
