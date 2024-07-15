package com.example.espark_1;

import android.os.Parcel;
import android.os.Parcelable;
import android.util.Pair;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

enum TYPE_STRUCTURE {sector, floor}
enum TYPE_VEHICLE {car, motorcycle, all}
enum TYPE {free, paid}
enum TYPE_LINE {white, green, blue, yellow, pink}

public class Parking implements Parcelable {

    private String _id;
    private String _name;
    private TYPE_STRUCTURE _typeStructure;
    private int _nSecFloor;
    private List<Parking_slot> _slot;
    private Pair<Float, Float> _coordinates;
    private Pair<Float, Float> _Endcoordinates;
    private TYPE _type;
    private TYPE_VEHICLE _typeVehicle;
    private List<TYPE_LINE> _typeLine;
    private List<Issue> _issues;
    private Float _costHourly;
    private String _streetName;

    public Parking(){}

    public Parking(String id, String name, TYPE_STRUCTURE typeStructure, int nSecFloor, List<Parking_slot> slot, Pair<Float, Float> coordinates, TYPE type,TYPE_VEHICLE typeVehicle, List<TYPE_LINE> typeLine, Float costHourly, Pair<Float, Float> Endcoordinates, String streetName/*, List<Issue> issue*/)
    {
        _id = id;
        _name = name;
        _typeStructure = typeStructure;
        _nSecFloor = nSecFloor;
        _slot = slot;
        _coordinates = coordinates;
        _type = type;
        _typeVehicle = typeVehicle;
        _typeLine = typeLine;
        _costHourly = costHourly;
        _Endcoordinates = Endcoordinates;
        _streetName = streetName;
        //_issues = issue;
    }

    protected Parking(Parcel in) {
        _id = in.readString();
        _name = in.readString();
        _typeStructure = TYPE_STRUCTURE.valueOf(in.readString());
        _nSecFloor = in.readInt();
        _slot = new ArrayList<>();
        in.readList(_slot, Parking_slot.class.getClassLoader());
        _coordinates = Pair.create(in.readFloat(), in.readFloat());
        _Endcoordinates = Pair.create(in.readFloat(), in.readFloat());
        _type = TYPE.valueOf(in.readString());
        _typeVehicle = TYPE_VEHICLE.valueOf(in.readString());
        _typeLine = new ArrayList<>();
        in.readList(_typeLine, TYPE_LINE.class.getClassLoader());
        _issues = new ArrayList<>();
        in.readList(_issues, Issue.class.getClassLoader());
        _costHourly = in.readFloat();
        _streetName = in.readString();
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(_id);
        dest.writeString(_name);
        dest.writeString(_typeStructure.name());
        dest.writeInt(_nSecFloor);
        dest.writeList(_slot);
        dest.writeFloat(_coordinates.first);
        dest.writeFloat(_coordinates.second);
        dest.writeFloat(_Endcoordinates.first);
        dest.writeFloat(_Endcoordinates.second);
        dest.writeString(_type.name());
        dest.writeString(_typeVehicle.name());
        dest.writeList(_typeLine);
        dest.writeFloat(_costHourly);
        dest.writeList(_issues);
        dest.writeString(_streetName);
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Creator<Parking> CREATOR = new Creator<Parking>() {
        @Override
        public Parking createFromParcel(Parcel in) {
            return new Parking(in);
        }

        @Override
        public Parking[] newArray(int size) {
            return new Parking[size];
        }
    };

    //getters
    public String id() { return _id; }
    public String name() { return _name; }
    public TYPE_STRUCTURE typeStructure() { return _typeStructure; }
    public int nSecFloor() { return _nSecFloor; }
    public List<Parking_slot> slot() { return _slot; }
    public Pair<Float, Float> coordinates() { return _coordinates; }
    public Pair<Float, Float> Endcoordinates() { return _Endcoordinates; }
    public TYPE type() { return _type; }
    public TYPE_VEHICLE typeVehicle() { return _typeVehicle; }
    public List<TYPE_LINE> typeLine() { return _typeLine; }
    public Float costHourly() { return _costHourly; }
    public String streetName() { return _streetName; }
    public List<Issue> issues() { return _issues; }

    //setters
    public void set_id(String id) { _id = id; }
    public void set_name(String name) { _name = name; }
    public void set_typeStructure(TYPE_STRUCTURE typeStructure) { _typeStructure = typeStructure; }
    public void set_nSecFloor(int nSecFloor) { _nSecFloor = nSecFloor; }
    public void set_slot(List<Parking_slot> slot) { _slot = slot; }
    public void set_coordinates(Pair<Float, Float> coordinates) { _coordinates = coordinates; }
    public void set_Endcoordinates(Pair<Float, Float> Endcoordinates) { _Endcoordinates = Endcoordinates; }
    public void set_type(TYPE type) { _type = type; }
    public void set_typeVehicle(TYPE_VEHICLE typeVehicle) { _typeVehicle = typeVehicle; }
    public void set_typeLine(List<TYPE_LINE> typeLine) { _typeLine = typeLine; }
    public void set_costHourly(Float costHourly) { _costHourly = costHourly; }
    public void set_issues(List<Issue> issues) { _issues = issues; }
    public void set_streetName(String streetName) { _streetName = streetName; }
}


