import re


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_var(elem, dataset='geo_prolog'):
    if dataset == 'geo_prolog':
        return elem.isupper() and len(elem) == 1
    elif dataset == 'job_prolog':
        return elem in ['ANS', 'X', 'A', 'B', 'P', 'J']
    elif dataset == 'geo_lambda' or dataset == 'atis_lambda':
        return elem.startswith('$')


def is_lit(elem, dataset='geo_prolog'):
    if dataset == 'geo_prolog':
        return elem.startswith('var')
    elif dataset == 'job_prolog':
        return elem.endswith('id0') or elem.endswith('id1') or elem.endswith('id2') or elem in ['20', 'hour',
                                                                                                'num_salary', 'year',
                                                                                                'year0', 'year1',
                                                                                                'month']
    elif dataset == 'geo_lambda':
        return ':ap' in elem or ':fb' in elem or ':mf' in elem or \
               ':me' in elem or ':cl' in elem or ':pd' in elem or \
               ':dc' in elem or ':al' in elem or \
               elem in ['yr0', 'do0', 'fb1', 'rc0', 'ci0', 'fn0', 'ap0', 'al1', 'al2', 'ap1', 'ci1',
                        'ci2', 'ci3', 'st0', 'ti0', 'ti1', 'da0', 'da1', 'da2', 'da3', 'da4', 'al0',
                        'fb0', 'dn0', 'dn1', 'mn0', 'ac0', 'fn1', 'st1', 'st2',
                        'c0', 'm0', 's0', 'r0', 'n0', 'co0', 'usa:co', 'death_valley:lo', 's1',
                        'colorado:n']
    elif dataset == 'atis_lambda':
        return ':ap' in elem or ':fb' in elem or ':mf' in elem or \
               ':me' in elem or ':cl' in elem or ':pd' in elem or \
               ':dc' in elem or ':al' in elem or \
               elem in ['yr0', 'do0', 'fb1', 'rc0', 'ci0', 'fn0', 'ap0', 'al1', 'al2', 'ap1', 'ci1',
                        'ci2', 'ci3', 'st0', 'ti0', 'ti1', 'da0', 'da1', 'da2', 'da3', 'da4', 'al0',
                        'fb0', 'dn0', 'dn1', 'mn0', 'ac0', 'fn1', 'st1', 'st2',
                        'c0', 'm0', 's0', 'r0', 'n0', 'co0', 'usa:co', 'death_valley:lo', 's1',
                        'colorado:n'] or elem.endswith(':i') or elem.endswith(':hr')

def is_predicate(elem, dataset='geo_prolog'):
    if dataset == 'geo_prolog':
        pass
    elif dataset == 'geo_lambda' or dataset == 'atis_lambda':
        return elem in ['jet', 'flight', 'from_airport', 'airport', 'airline', 'airline_name',
                        'class_type', 'aircraft_code', 'aircraft_code:t',
                        'from', 'to', 'day', 'month', 'year', 'arrival_time', 'limousine',
                        'departure_time', 'meal', 'meal:t', 'meal_code',
                        'during_day', 'tomorrow', 'daily', 'time_elapsed', 'time_zone_code',
                        'booking_class:t', 'booking_class', 'economy', 'ground_fare', 'class_of_service',
                        'capacity', 'weekday', 'today', 'turboprop', 'aircraft', 'air_taxi_operation',
                        'month_return', 'day_return', 'day_number_return', 'minimum_connection_time',
                        'during_day_arrival', 'connecting', 'minutes_distant',
                        'named', 'miles_distant', 'approx_arrival_time', 'approx_return_time',
                        'approx_departure_time', 'has_stops',
                        'day_after_tomorrow', 'manufacturer', 'discounted', 'overnight',
                        'nonstop', 'has_meal', 'round_trip', 'oneway', 'loc:t', 'ground_transport',
                        'to_city', 'flight_number', 'equals:t', 'abbrev', 'equals', 'rapid_transit',
                        'stop_arrival_time', 'arrival_month', 'cost',
                        'fare', 'services', 'fare_basis_code', 'rental_car', 'city', 'stop', 'day_number',
                        'days_from_today', 'after_day', 'before_day',
                        'airline:e', 'stops', 'month_arrival', 'day_number_arrival', 'day_arrival', 'taxi',
                        'next_days', 'restriction_code', 'tomorrow_arrival', 'tonight',
                        'population:i', 'state:t', 'next_to:t', 'elevation:i', 'size:i', 'capital:t',
                        'len:i', 'city:t', 'named:t', 'river:t', 'place:t', 'capital:c', 'major:t', 'town:t',
                        'mountain:t', 'lake:t', 'area:i', 'density:i', 'high_point:t', 'elevation:t', 'population:t',
                        'in:t']
    elif dataset == 'job_prolog':
        return elem in ['job', 'language', 'loc', 'req_deg', 'application', 'area', 'company',
                        'des_deg', 'des_exp', 'platform', 'recruiter', 'req_exp', 'salary_greater_than',
                        'salary_less_than', 'title', 's']


def rm_subbag(bag_to_check, super_bag):
    for elem in bag_to_check:
        if elem in super_bag:
            super_bag.remove(elem)
        else:
            raise Exception('{} is not in {}'.format(elem, super_bag))
