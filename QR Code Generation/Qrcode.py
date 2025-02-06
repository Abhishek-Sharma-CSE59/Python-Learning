import qrcode
import json

def generate_qr_code(data, filename="person_data.png"):
    """Generates a QR code from the given data and saves it to a file."""

    qr = qrcode.QRCode(
        version=1,  # Adjust version as needed for data size
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Adjust error correction as needed
        box_size=10,  # Adjust box size as needed
        border=4,  # Adjust border size as needed
    )

    qr.add_data(data)
    qr.make(fit=True)  # Make the QR code fit the data

    img = qr.make_image(fill_color="black", back_color="white") # Customize colors
    img.save(filename)
    print(f"QR code saved to {filename}")


def main():
    """Main function to handle data input and QR code generation."""

    person_data = {
        "name": input("Enter name: "),
        "age": int(input("Enter age: ")), #convert it into integer
        "city": input("Enter city: "),
        "email": input("Enter email: ")
    }

    # Convert the dictionary to JSON string
    json_data = json.dumps(person_data)

    generate_qr_code(json_data)


if __name__ == "__main__":
    main()