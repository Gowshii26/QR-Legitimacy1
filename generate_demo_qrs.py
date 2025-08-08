import qrcode
import os

def generate_demo_qrs():
    output_dir = "qr_images"
    os.makedirs(output_dir, exist_ok=True)
    
    demo_qrs = [
        {"filename": "legit_upi.png", "content": "upi://pay?pa=afzal@upi&pn=Afzal%20Ahamed&am=50.25&tn=Groceries"},
        {"filename": "scam_upi.png", "content": "upi://pay?pa=fraud123@unknown&pn=Verify%20Account&am=1000.00&tn=Urgent%20Payment"}
    ]
    
    for qr in demo_qrs:
        qr_code = qrcode.QRCode(version=1, box_size=10, border=4)
        qr_code.add_data(qr["content"])
        qr_code.make(fit=True)
        img = qr_code.make_image(fill="black", back_color="white")
        img.save(os.path.join(output_dir, qr["filename"]))
        print(f"Generated {qr['filename']}")

if __name__ == "__main__":
    generate_demo_qrs()