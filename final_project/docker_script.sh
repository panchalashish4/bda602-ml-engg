#!/bin/bash

sleep 60

if mariadb -u root -ppassword123 -h mariadb5 -e "USE baseball;"
then
    echo "Database exist, creating feature table.."
    mariadb -u root -ppassword123 -h mariadb5 baseball < features_table.sql
    echo "Feature table successfully created."
else
    echo "Database does not exist, loading database.."
    mariadb -u root -ppassword123 -h mariadb5 -e "CREATE DATABASE baseball;"
    mariadb -u root -ppassword123 -h mariadb5 baseball < baseball.sql
    echo "Database Loaded, creating feature table.."
    mariadb -u root -ppassword123 -h mariadb5 baseball < features_table.sql
    echo "Feature table successfully created."
fi

python3 main.py
