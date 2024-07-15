package com.example.espark_1;


import static android.content.ContentValues.TAG;

import android.util.Log;
import android.util.Pair;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.messaging.FirebaseMessaging;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class User {
    private static String _id;
    private static String _UID;
    private static String _email;
    //private static String _password;
    private static String _name;
    private static String _surname;
    private static Integer _age;
    //private static String _telNumber;
    private static List<Pair<String, String>> _payMethod;
    private static List<Pair<String, String>> _vehicle;
    private static List<Pair<Pair<String, String>, String>> _selectedSlot;
    private static List<Pair<String, String>> _favouriteParking;
    private static List<ParkingStop> _ongoing;
    private static List<ParkingStop> _history;
    private static String _notificationToken;

    public User(){}

    public User(String id, String UID, String email/*, String password*/, String name, String surname, Integer age/*, String telNumber*/, String notificationToken)
    {
        id = _id;
        _UID = UID;
        _email = email;
        //_password = password;
        _name = name;
        _surname = surname;
        _age = age;
        //_telNumber = telNumber;
        _notificationToken = notificationToken;
        //add other variables//
    }

    //getters
    public static String id() { return _id; }
    public static String UID() { return _UID; }
    public static String email() { return _email; }
    //public static String password() { return _password; }
    public static String name() { return _name; }
    public static String surname() { return _surname; }
    public static Integer age() { return _age; }
    //public static String telNumber() { return _telNumber; }
    public static List<Pair<String, String>> payMethod() { return _payMethod; }
    public static List<Pair<String, String>> vehicle() { return _vehicle; }
    public static List<Pair<Pair<String, String>, String>> selectedSlot() { return _selectedSlot; }
    public static List<Pair<String, String>> favouriteParking() { return _favouriteParking; }
    public static List<ParkingStop> ongoing() { return _ongoing; }
    public static List<ParkingStop> history() { return _history; }
    public static String notificationToken() { return _notificationToken; }



    //setters
    public static void set_id(String id) { _id = id; }
    public static void set_uid(String UID) { _UID = UID; }
    public static void set_email(String email) { _email = email; }
    //public static void set_password(String password) { _password = password; }
    public static void set_name(String name) { _name = name; }
    public static void set_surname(String surname) { _surname = surname; }
    public static void set_age(Integer age) { _age = age; }
    //public static void set_telNumber(String telNumber) { _telNumber = telNumber; }
    public static void set_payMethod(List<Pair<String, String>> payMethod) { _payMethod = payMethod; }
    public static void set_vehicle(List<Pair<String, String>> vehicle) { _vehicle = vehicle; }
    public static void set_selectedSlot(List<Pair<Pair<String, String>, String>> selectedSlot) { _selectedSlot = selectedSlot; }
    public static void set_favouriteParking(List<Pair<String, String>> favouriteParking) { _favouriteParking = favouriteParking; }
    public static void set_ongoing(List<ParkingStop> ongoing) { _ongoing = ongoing; }
    public static void set_history(List<ParkingStop> history) { _history = history; }
    public static void set_notificationToken(String notificationToken) { _notificationToken = notificationToken; }

    //update
    public static void slotUpdateDB()
    {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("user").child(User.id()).child("selectedSlot");
        myRef.removeValue();

        for (int i = 0; i < User.selectedSlot().size(); i++)
        {
            myRef.child("s" + String.valueOf(i + 1)).child("parkingId").setValue(User.selectedSlot().get(i).second);
            myRef.child("s" + String.valueOf(i + 1)).child("parkingName").setValue(User.selectedSlot().get(i).first.first);
            myRef.child("s" + String.valueOf(i + 1)).child("slotNum").setValue(User.selectedSlot().get(i).first.second);
        }
    }
    public static void vehicleUpdateDB()
    {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("user").child(User.id()).child("vehicle");
        myRef.removeValue();

        for (int i = 0; i < User.vehicle().size(); i++)
        {
            myRef.child("car" + String.valueOf(i + 1)).child("license").setValue(User.vehicle().get(i).first);
            myRef.child("car" + String.valueOf(i + 1)).child("model").setValue(User.vehicle().get(i).second);
        }
    }
    public static void cardUpdateDB()
    {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("user").child(User.id()).child("methodPay");
        myRef.removeValue();

        for (int i = 0; i < User.payMethod().size(); i++)
        {
            myRef.child("mp" + String.valueOf(i + 1)).child("numCard").setValue(User.payMethod().get(i).first);
            myRef.child("mp" + String.valueOf(i + 1)).child("type").setValue(User.payMethod().get(i).second);
        }
    }

    public static void tokenUpdateDB()
    {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("user").child(User.id()).child("token");
        FirebaseMessaging.getInstance().getToken()
                .addOnCompleteListener(new OnCompleteListener<String>() {
                    @Override
                    public void onComplete(@NonNull Task<String> task) {
                        if (!task.isSuccessful()) {
                            //Log.w(TAG, "Fetching FCM registration token failed", task.getException());
                            return;
                        }

                        // Get new FCM registration token
                        String token = task.getResult();
                        myRef.setValue(token);
                    }
                });
    }
    public static void userUpdate()
    {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("user");

        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String id = childSnapshot.getKey();
                    String userUID = childSnapshot.child("uid").getValue(String.class);
                    if (userUID.equals(_UID)) {
                        User.set_id(id);
                        User.set_uid(userUID);
                        //Log.d("LOGIN PAGE 2", String.valueOf(User._UID));
                        User.set_name(childSnapshot.child("name").getValue(String.class));
                        //Log.d("LOGIN PAGE 2", String.valueOf(User.name()));
                        User.set_surname(childSnapshot.child("surname").getValue(String.class));
                        User.set_email(childSnapshot.child("email").getValue(String.class));
                        //User.set_password(childSnapshot.child("password").getValue(String.class));
                        User.set_age(childSnapshot.child("age").getValue(Integer.class));
                        //User.set_telNumber(childSnapshot.child("telNumber").getValue(String.class));

                        List<Pair<String, String>> vehicleList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("vehicle").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            String l = lineSnapshot.child("license").getValue(String.class);
                            String m = lineSnapshot.child("model").getValue(String.class);
                            Pair<String, String> v = new Pair<>(l, m);
                            //Log.d("LOGIN PAGE 2", String.valueOf(l));
                            vehicleList.add(v);
                        }
                        User.set_vehicle(vehicleList);

                        List<Pair<String, String>> payList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("methodPay").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> p = new Pair<>(lineSnapshot.child("numCard").getValue(String.class), lineSnapshot.child("type").getValue(String.class));
                            payList.add(p);
                        }
                        User.set_payMethod(payList);

                        List<Pair<Pair<String, String>, String>> slotList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("selectedSlot").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> s = new Pair<>(lineSnapshot.child("parkingName").getValue(String.class), lineSnapshot.child("slotNum").getValue(String.class));
                            Pair<Pair<String, String>, String> s1 = new Pair<>(s,lineSnapshot.child("parkingId").getValue(String.class));
                            slotList.add(s1);
                        }
                        User.set_selectedSlot(slotList);

                        List<Pair<String, String>> favouriteList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("favouriteParking").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> s = new Pair<>(a, lineSnapshot.child("name").getValue(String.class));
                            favouriteList.add(s);
                        }
                        User.set_favouriteParking(favouriteList);

                        List<ParkingStop> ongoingList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("ongoingParking").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            ParkingStop ps = new ParkingStop();
                            ps.set_id(a);
                            ps.set_name(lineSnapshot.child("parkingName").getValue(String.class));
                            ps.set_prize(lineSnapshot.child("prize").getValue(Float.class));
                            ps.set_vehicleLP(lineSnapshot.child("vehicleLP").getValue(String.class));
                            ps.set_payCard(lineSnapshot.child("payCard").getValue(String.class));
                            ps.set_costHourly(lineSnapshot.child("costH").getValue(Float.class));
                            SimpleDateFormat format=new SimpleDateFormat("dd-MM-yyyy HH:mm");
                            String d = lineSnapshot.child("startDate").getValue(String.class);
                            //Log.d("LOGIN PAGE 2", String.valueOf(d));
                            try {
                                Date date = format.parse(d);
                                //System.out.println("Parsed Date-Time: " + date);
                                ps.set_startDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            String d1 = lineSnapshot.child("endDate").getValue(String.class);
                            //Log.d("LOGIN PAGE 2", String.valueOf(d1));
                            try {
                                Date date = format.parse(d1);
                                //System.out.println("Parsed Date-Time: " + date);
                                ps.set_endDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            ongoingList.add(ps);

                        }
                        User.set_ongoing(ongoingList);

                        List<ParkingStop> historyList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("parkingHistory").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            ParkingStop ps = new ParkingStop();
                            ps.set_id(a);
                            ps.set_name(lineSnapshot.child("parkingName").getValue(String.class));
                            ps.set_prize(lineSnapshot.child("prize").getValue(Float.class));
                            SimpleDateFormat format=new SimpleDateFormat("dd-MM-yyyy HH:mm");
                            String d = lineSnapshot.child("startDate").getValue(String.class);
                            try {
                                Date date = format.parse(d);

                                ps.set_startDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            String d1 = lineSnapshot.child("endDate").getValue(String.class);
                            try {
                                Date date = format.parse(d1);
                                ps.set_endDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            historyList.add(ps);

                        }
                        User.set_history(historyList);

                        break;
                    }
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        });

    }
}
