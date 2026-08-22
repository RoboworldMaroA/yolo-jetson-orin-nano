
# from adas import app, producer, stop_event
#Add certificat that allow me to use microfone on the phone and computer using safe connection
# Acces to the app thrue the https://localhost
# from adas_road import app, producer, stop_event
from adas_road_voice import app, producer, stop_event
import threading

if __name__ == "__main__":
    prod = threading.Thread(
        target=producer,
        daemon=True
    )

    prod.start()

    try:
        app.run(
            host="0.0.0.0",
            port=5010,
            threaded=True,
            ssl_context=('/app/cert.pem', '/app/key.pem')
        )
    finally:
        stop_event.set()
        prod.join(timeout=2)