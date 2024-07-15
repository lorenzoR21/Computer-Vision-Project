package com.example.espark_1;

import android.os.Parcelable;
import android.os.Parcel;
import android.util.Pair;

import java.util.ArrayList;
import java.util.List;

public class Parking_slot implements Parcelable
{
    private String _id;
    private String _value;
    private TYPE_LINE _type;

    private int _status;

    public Parking_slot() {}

    public Parking_slot(String id, String value, TYPE_LINE type, int status)
    {
        _id = id;
        _value = value;
        _type = type;
        _status = status;
    }

    protected Parking_slot(Parcel in) {
        _id = in.readString();
        _value = in.readString();
        _type = TYPE_LINE.valueOf(in.readString());;
        _status = in.readInt();
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(_id);
        dest.writeString(_value);
        dest.writeString(_type.name());
        dest.writeInt(_status);
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Creator<Parking_slot> CREATOR = new Creator<Parking_slot>() {
        @Override
        public Parking_slot createFromParcel(Parcel in) {
            return new Parking_slot(in);
        }

        @Override
        public Parking_slot[] newArray(int size) {
            return new Parking_slot[size];
        }
    };

    //getters
    public String id() { return _id; }
    public String value() { return _value; }
    public TYPE_LINE type() { return _type; }
    public int status() { return _status; }

    //setters
    public void set_id(String id) { _id = id; }
    public void set_value(String value) { _value = value; }
    public void set_type(TYPE_LINE type) { _type = type; }
    public void set_status(int status) { _status = status; }
}
