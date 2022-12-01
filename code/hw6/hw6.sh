#!/bin/bash

sleep 180

if mariadb -u root -ppassword123 -h mariadb3 -e "USE baseball;"
then
    echo "Database exist"
    mariadb -u root -ppassword123 -h mariadb3 baseball < hw6.sql | tee ./files/result.csv
else
    echo "Database does not exist, loading database.."
    mariadb -u root -ppassword123 -h mariadb3 -e "CREATE DATABASE baseball;"
    mariadb -u root -ppassword123 -h mariadb3 baseball < baseball.sql
    mariadb -u root -ppassword123 -h mariadb3 baseball < hw6.sql | tee ./files/result.csv
fi
