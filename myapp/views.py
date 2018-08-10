from django.shortcuts import render
import datetime

# Create your views here.
from django.http import HttpResponse

def hello(request):
   today =datetime. datetime.now().date()
   daysOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

   return render(request, "hello.html", {'today': today, "days_of_week" : daysOfWeek})

# Interacting directly with a model (CRUD OPERATIONS)
from myapp.models import Contacts
from django.http import HttpResponse


def crudops(request):
   # Creating an entry

   contacts = Contacts(
      website="www.polo.com", mail="sorex@polo.com",
      name="sorex", phonenumber="002376970"
   )

   contacts.save()

   # Read ALL entries
   objects = Contacts.objects.all()
   res = 'Printing all Dreamreal entries in the DB : <br>'

   for elt in objects:
      res += elt.name + "<br>"

   # Read a specific entry:
   sorex = Contacts.objects.get(name="sorex")
   res += 'Printing One entry <br>'
   res += sorex.name

   # Delete an entry
   res += '<br> Deleting an entry <br>'
   sorex.delete()

   # Update
   contacts = Contacts(
      website="www.polo.com", mail="sorex@polo.com",
      name="sorex", phonenumber="002376970"
   )

   contacts.save()
   res += 'Updating entry<br>'

   contacts = Contacts.objects.get(name='sorex')
   contacts.name = 'thierry'
   contacts.save()

   return HttpResponse(res)

# Interacting with the class representing our model (
# CRUD OPERATIONS)

def datamanipulation(request):
   res = ''

   # Filtering data:
   qs = Contacts.objects.filter(name="paul")
   res += "Found : %s results<br>" % len(qs)

   # Ordering results
   qs = Contacts.objects.order_by("mail")

   for elt in qs:
      res += elt.name + '<br>'

   return HttpResponse(res)
