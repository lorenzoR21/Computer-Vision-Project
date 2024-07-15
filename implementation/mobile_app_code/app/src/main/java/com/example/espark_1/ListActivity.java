package com.example.espark_1;

import android.os.Bundle;

import com.example.espark_1.databinding.ActivityListBinding;

public class ListActivity extends DrawerBaseActivity {

    private ActivityListBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
    }
}