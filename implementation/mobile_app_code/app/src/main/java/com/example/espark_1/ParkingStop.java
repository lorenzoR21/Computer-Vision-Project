package com.example.espark_1;

import java.time.LocalDateTime;
import java.util.Date;

public class ParkingStop
{
    private String _id;
    private String _name;
    private Float _prize;
    private Date _startDate;
    private Date _endDate;
    private String _vehicleLP;
    private String _payCard;
    private Float _costHourly;
    private String _ongoing;

    public ParkingStop(){}

    public ParkingStop(String id, String name, Float prize, Date startDate, Date endDate, String vehicleLP, String payCard, Float costHourly, String ongoing)
    {
        _id = id;
        _name = name;
        _prize = prize;
        _startDate = startDate;
        _endDate = endDate;
        _vehicleLP = vehicleLP;
        _payCard = payCard;
        _costHourly = costHourly;
        _ongoing = ongoing;
    }

    //getters
    public String id() { return _id; }
    public String name() { return _name; }
    public Float prize() { return _prize; }
    public Date startDate() { return _startDate; }
    public Date endDate() { return _endDate; }
    public String vehicleLP() { return _vehicleLP; }
    public String payCard() { return _payCard; }
    public Float costHourly() { return _costHourly; }
    public String ongoing() { return _ongoing; }




    //setters
    public void set_id(String id) { _id = id; }
    public void set_name(String name) { _name = name; }
    public void set_prize(Float prize) { _prize = prize; }
    public void set_startDate(Date startDate) { _startDate = startDate; }
    public void set_endDate(Date endDate) { _endDate = endDate; }
    public void set_vehicleLP(String vehicleLP) { _vehicleLP = vehicleLP; }
    public void set_payCard(String payCard) { _payCard = payCard; }
    public void set_costHourly(Float costHourly) { _costHourly = costHourly; }
    public void set_ongoing(String ongoing) { _ongoing = ongoing; }

}
