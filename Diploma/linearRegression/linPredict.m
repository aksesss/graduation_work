function [ y ] = linPredict( X, theta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
y = [ones(size(X,1), 1) X]*theta;

end