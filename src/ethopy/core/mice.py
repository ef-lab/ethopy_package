"""Create tables for the lab_mice schema.

Mirrors the animal colony schema hosted at `lab_mice`: animal identity, lines and
genotypes, weights, surgeries/implants and location transfers. Importing this
module declares any missing table in the schema mapped to ``mice`` in SCHEMATA.
"""

import datajoint as dj

from ethopy.core.logger import mice  # noqa: F401


@mice.schema
class Lines(dj.Manual):
    """Basic mouse line info."""

    definition = """
    # Basic mouse line info
    line                 : varchar(100)                 # Mouse Line Abbreviation
    ---
    line_full            : varchar(100)                 # full line name
    rec_strain           : varchar(20)                  # recipient strain
    donor_strain         : varchar(20)                  # donor strain
    n=null               : tinyint                      # minimum number of backcrosses to recipient strain
    seq                  : varchar(5000)                # sequence of transgene, if available
    line_notes           : varchar(4096)                # other comments
    line_ts=CURRENT_TIMESTAMP : timestamp               # automatic
    """


@mice.schema
class Mice(dj.Manual):
    """Basic mouse info."""

    definition = """
    # Basic mouse info
    animal_id            : int                          # id number
    ---
    other_id=''          : varchar(20)                  # alternative id number
    dob=null             : date                         # animal's date of birth
    dow=null             : date                         # animal's date of weaning
    sex='unknown'        : enum('M','F','unknown')      # animal's sex
    color='unknown'      : enum('Black','Brown','White','unknown') # animal's color
    line=''              : varchar(255)                 # mouse line
    genotype='unknown'   : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','unknown')
    ear_punch='unknown'  : enum('None','R','L','RL','RR','LL','unknown') # animal's ear punch
    owner='none'         : enum('manolis','maria','emina','Other','Available','none') # mouse's owner
    fluo_test='unknown'  : enum('unknown','no','yes')   # fluorescence test result
    mouse_notes=''       : varchar(4096)                # other comments and distinguishing features
    facility='unknown'   : enum('TMF','Taub','Other','unknown') # animal's current facility
    room='unknown'       : enum('VK3','VH1','T014','T057','T086D','Other','unknown','T027') # animal's current room
    rack=null            : char(4)                      # animal's current rack
    row=''               : char(1)                      # animal's current row
    mouse_ts=CURRENT_TIMESTAMP : timestamp              # automatic
    cage_id=''           : varchar(100)                 # animal's current cage
    usage='unknown'      : enum('in use','available','euthanize','unknown') # availability
    """


@mice.schema
class Death(dj.Manual):
    """Info about each mouse's death."""

    definition = """
    # info about each mouse's death
    -> Mice
    ---
    dod=null             : date                         # date of death
    death_notes=''       : varchar(4096)                # other comments
    death_ts=CURRENT_TIMESTAMP : timestamp              # automatic
    """


@mice.schema
class Founders(dj.Manual):
    """Additional info about founder mice."""

    definition = """
    # Additional info about founder mice
    -> Mice
    -> Lines
    ---
    source               : varchar(100)                 # source of mouse (lab, company)
    doa=null             : date                         # date of arrival
    founder_notes        : varchar(4096)                # other comments
    founder_ts=CURRENT_TIMESTAMP : timestamp            # automatic
    """


@mice.schema
class Genotypes(dj.Manual):
    """Info about each mouse's genotype."""

    definition = """
    # info about each mouse's genotype
    -> Mice
    -> Lines
    ---
    genotype='unknown'   : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','unknown') # animal's genotype
    genotype_notes=null  : varchar(4096)                # other comments
    genotype_ts=CURRENT_TIMESTAMP : timestamp           # automatic
    """


@mice.schema
class Parents(dj.Manual):
    """Parent-child relationships between mice."""

    definition = """
    # parent-child relationships between mice
    -> Mice
    parent_id            : varchar(20)                  # id number of parent
    ---
    relation_notes=''    : varchar(4096)                # other comments
    relation_ts=CURRENT_TIMESTAMP : timestamp           # automatic
    """


@mice.schema
class Transfers(dj.Manual):
    """Completed transfers."""

    definition = """
    # completed transfers
    -> Mice
    dot                  : date                         # date of transfer
    ---
    from_owner='none'    : enum('alex','manolis','Other','Available','none') # previous owner
    to_owner='none'      : enum('alex','manolis','Other','Available','none') # new owner
    from_facility='unknown' : enum('TMF','Taub','Other','unknown') # animal's previous facility
    to_facility='unknown' : enum('TMF','Taub','Other','unknown')   # animal's new facility
    from_room='unknown'  : enum('VD4','T014','T057','T086D','Other','unknown','VK3','T027','VH1') # animal's previous room
    to_room='unknown'    : enum('VD4','T014','T057','T086D','Other','unknown','VK3','T027','VH1') # animal's new room
    from_rack=null       : char(4)                      # animal's previous rack
    to_rack=null         : char(4)                      # animal's new rack
    from_row=''          : char(1)                      # animal's previous row
    to_row=''            : char(1)                      # animal's new row
    transfer_notes=''    : varchar(4096)                # other comments
    transfer_ts=CURRENT_TIMESTAMP : timestamp           # automatic
    """


@mice.schema
class MouseWeight(dj.Manual):
    """Weight measurements, logged by the setup at session start."""

    definition = """
    animal_id            : int unsigned                 # id number
    timestamp=CURRENT_TIMESTAMP : timestamp             # timestamp of weight
    ---
    weight               : double(5,2)                  # weight in grams
    """


@mice.schema
class GrowthCurve(dj.Lookup):
    """Reference weight per age, sex and genotype."""

    definition = """
    age                  : int                          # in weeks
    gender               : enum('male','female')
    genotype             : enum('C57BL/6J')
    ---
    weight=null          : double                       # in grams
    std=null             : double                       # standard deviation
    """


@mice.schema
class SurgeryType(dj.Lookup):
    """Surgery types."""

    definition = """
    # Surgery types
    surgery              : varchar(16)                  # aim
    ---
    description=''       : varchar(2048)                # description
    """


@mice.schema
class Surgery(dj.Manual):
    """Surgery information."""

    definition = """
    # Surgery information
    animal_id            : smallint unsigned            # animal id
    timestamp            : datetime                     # timestamp
    ---
    user_name            : varchar(16)                  # user performing the surgery
    -> SurgeryType
    note=null            : varchar(2048)                # surgery notes
    """


@mice.schema
class Implants(dj.Manual):
    """Implant information."""

    definition = """
    animal_id            : int unsigned                 # id number
    doi                  : date                         # date of implantation
    ---
    experimenter='Other' : varchar(64)                  # name of experimenter
    anesthesia='Other'   : enum('isoflurane','ketamine/xylazine mix','Other') # anesthesia method
    comments=null        : varchar(100)
    """


@mice.schema
class Handling(dj.Manual):
    """Handling sessions."""

    definition = """
    animal_id            : int unsigned                 # id number
    timestamp            : datetime                     # date of handling
    ---
    experimenter='Other' : varchar(64)                  # name of experimenter
    type='Touch'         : enum('Touch','Other')        # handling method
    comments=null        : varchar(100)
    """


@mice.schema
class Person(dj.Manual):
    """People in the lab."""

    definition = """
    # people in the lab
    person               : varchar(12)                  # person's short name
    ---
    full_name            : varchar(64)                  # person's full name
    """
