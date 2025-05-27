import streamlit as st
import datetime
import pandas as pd

st.set_page_config(page_title="Hospital Management System", layout="centered")
st.title("🏥 Hospital Management System")

def load_doctors():
    with open("doctors.txt", "r") as f:
        return [line.strip().split(",") for line in f.readlines()]

def save_patient(name, age, gender, contact):
    with open("patients.txt", "a") as f:
        f.write(f"{name},{age},{gender},{contact},{datetime.date.today()}\n")

def save_appointment(patient, doctor, time):
    with open("appointments.txt", "a") as f:
        f.write(f"{patient},{doctor},{time},{datetime.date.today()}\n")

def save_prescription(patient, prescription):
    with open("prescriptions.txt", "a") as f:
        f.write(f"{patient},{prescription},{datetime.date.today()}\n")

def read_file(file):
    with open(file, "r") as f:
        return [line.strip().split(",") for line in f.readlines()]

menu = st.sidebar.radio("Go to", ["Patient Registration", "Book Appointment", "Prescriptions", "Daily Report"])

if menu == "Patient Registration":
    st.header("📝 Register Patient")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    contact = st.text_input("Contact Number")

    if st.button("Register"):
        if name and contact:
            save_patient(name, age, gender, contact)
            st.success("Patient registered successfully!")
        else:
            st.error("Please fill all fields.")

elif menu == "Book Appointment":
    st.header("📅 Book Appointment")
    patient_name = st.text_input("Enter Your Name")
    doctors = load_doctors()
    doctor_names = [f"{doc[0]} ({doc[1]}) - {doc[2]}" for doc in doctors]
    choice = st.selectbox("Select Doctor & Timing", doctor_names)

    if st.button("Confirm Appointment"):
        if patient_name:
            doctor = choice.split(" (")[0]
            time = choice.split("-")[-1].strip()
            save_appointment(patient_name, doctor, time)
            st.success(f"Appointment booked with {doctor} at {time}")
        else:
            st.error("Enter your name before booking.")

elif menu == "Prescriptions":
    st.header("💊 Add Prescription")
    pname = st.text_input("Patient Name")
    pres = st.text_area("Enter Prescription Details")

    if st.button("Save Prescription"):
        if pname and pres:
            save_prescription(pname, pres)
            st.success("Prescription saved successfully!")
        else:
            st.error("Both fields are required.")

elif menu == "Daily Report":
    st.header("📋 Daily Report")
    today = str(datetime.date.today())

    def filter_today(data):
        return [row for row in data if row[-1] == today]

    patients = filter_today(read_file("patients.txt"))
    appointments = filter_today(read_file("appointments.txt"))
    with open("prescriptions.txt", "r") as f:
     prescriptions_raw = f.readlines()
     prescriptions = []
    for line in prescriptions_raw:
     parts = line.strip().split(",", 2)
    if len(parts) == 3 and parts[2] == str(datetime.date.today()):
     prescriptions.append(parts)

    st.subheader("🧑‍🤝‍🧑 Patients Registered Today")
    st.dataframe(pd.DataFrame(patients, columns=["Name", "Age", "Gender", "Contact", "Date"]))

    st.subheader("📅 Appointments Today")
    st.dataframe(pd.DataFrame(appointments, columns=["Patient", "Doctor", "Time", "Date"]))

    st.subheader("💊 Prescriptions Today")
    with open("prescriptions.txt", "r") as f:
     prescriptions_raw = f.readlines()

    prescriptions = []
    for line in prescriptions_raw:
     parts = line.strip().split(",", 2)
    if len(parts) == 3 and parts[2] == str(datetime.date.today()):
        prescriptions.append(parts)

    st.dataframe(pd.DataFrame(prescriptions, columns=["Patient", "Prescription", "Date"]))

    if st.button("Export Report"):
        df_patients = pd.DataFrame(patients, columns=["Name", "Age", "Gender", "Contact", "Date"])
        df_appointments = pd.DataFrame(appointments, columns=["Patient", "Doctor", "Time", "Date"])
        df_prescriptions = pd.DataFrame(prescriptions, columns=["Patient", "Prescription", "Date"])

        df_patients.to_csv(f"daily_reports/patients_{today}.csv", index=False)
        df_appointments.to_csv(f"daily_reports/appointments_{today}.csv", index=False)
        df_prescriptions.to_csv(f"daily_reports/prescriptions_{today}.csv", index=False)

        st.success("Reports exported to 'daily_reports/' folder!")